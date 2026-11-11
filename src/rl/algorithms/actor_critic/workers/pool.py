import random
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Semaphore
from queue import Empty

import torch
import torch.multiprocessing as mp

from src.rl.actions import engine_action
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.algorithms.actor_critic.envs import new_env
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.batches import allocate_states
from src.rl.batches import cat_states
from src.rl.batches import count_jagged_rows
from src.rl.batches import iter_tensors
from src.rl.batches import map_tensors
from src.rl.batches import read_states
from src.rl.batches import to_device
from src.rl.encoding import encode_states_into
from src.rl.reward import RewardStream
from src.rl.types import StateBatch
from src.rl.types import StateBatchReserved
from src.rl.utils import parent_alive
from src.rl.utils import set_rng_seeds


@dataclass
class _StepBuffer:
    """One worker's shared memory: every tensor it and the Actor exchange each step.

    The token and sparse arrays are sized for the worst case, and the encoders write their
    leading rows, state after state. `read_states` generates the Jagged records the model
    consumes from those rows; the rows past them are stale. Rewriting a buffer invalidates
    earlier reads of it.
    """

    # One row per game, written by the Actor
    x_actions: torch.Tensor  # (B, 3) long: action type, state-local token offset, Monster target rank
    x_ascensions: torch.Tensor  # (B,) long. Each game's ascension under the live position

    # One row per game, written by the worker
    x_rewards: torch.Tensor  # (B, K)
    x_dones: torch.Tensor  # (B,)

    states: StateBatchReserved  # The encoded states, at worst-case capacity


@dataclass
class _WorkerHandle:
    """One worker process and its semaphores (the step loop's request, its reply)."""

    process: BaseProcess
    sph_request: Semaphore
    sph_reply: Semaphore


@dataclass
class Workers:
    """The Actor's handle on its environment workers.

    Pass it to `step_workers`, `write_ascensions`, `snapshot_to_device` and `stop_workers`. The
    worker package owns its shared storage and processes.
    """

    _step_buffers: list[_StepBuffer]  # One per worker, in game order
    _handles: list[_WorkerHandle]
    _episode_result_queue: Queue[EpisodeResult]
    _x_stop: torch.Tensor  # () bool in shared memory


def start_workers(
    num_envs: int, num_workers: int, ascension_current: float, gamma: float, seed: int
) -> Workers:
    """Start `num_workers` processes, each stepping an equal share of the `num_envs` games.

    It then waits until every one has written its games' first states.

    Parameters
    ----------
    num_envs : int
        The number of games; they must divide evenly across the workers.
    num_workers : int
        The number of worker processes.
    ascension_current : float
        The soft-curriculum position the games start at.
    gamma : float
        The discount the games' rewards and discounted returns use.
    seed : int
        The seed each worker's own seed derives from.

    Returns
    -------
    Workers
        The handle to pass to the worker operations, with the initial states ready to snapshot.
    """
    if num_envs <= 0 or num_workers <= 0:
        raise ValueError("`num_envs` and `num_workers` must be positive")
    
    if num_envs % num_workers != 0:
        raise RuntimeError("`num_envs` must divide evenly across workers")

    # Set spawn context. CUDA is live in this process; fork is unsafe
    ctx = mp.get_context("spawn") 

    # Instance `StepBuffer`s
    envs_per_worker = num_envs // num_workers
    step_buffers = _allocate_step_buffers(num_workers, envs_per_worker)

    # Boolean flag to signal shutdown
    x_stop = torch.tensor(False).share_memory_()

    workers = Workers(step_buffers, [], ctx.Queue(), x_stop)

    # Write the curriculum's current ascension level
    write_ascensions(workers, ascension_current)

    try:
        for idx_step_buffer, step_buffer in enumerate(step_buffers):
            # Semaphores: an Event.set() blocks until its sleepers wake; a killed worker hangs it
            sph_request, sph_reply = ctx.Semaphore(0), ctx.Semaphore(0)

            # Start worker process
            process = ctx.Process(
                target=_worker_loop,
                args=(
                    step_buffer,
                    workers._episode_result_queue,
                    sph_request,
                    sph_reply,
                    x_stop,
                    seed * 7919 + idx_step_buffer,
                    gamma,
                ),
                daemon=True,
            )
            process.start()

            # Append worker handle
            workers._handles.append(_WorkerHandle(process, sph_request, sph_reply))

        # Wait for all Workers to reply and write initial states
        _wait_for_replies(workers._handles)

    except BaseException:
        stop_workers(workers)
        raise

    return workers


def _allocate_step_buffers(num_buffers: int, buffer_size: int) -> list[_StepBuffer]:
    """Allocate reusable shared CPU memory and return one view of it per buffer.

    Parameters
    ----------
    num_buffers : int
        The number of independent buffers.
    buffer_size : int
        The number of states written into each buffer.

    Returns
    -------
    list[_StepBuffer]
        One buffer per index, each viewing its row of the shared CPU tensors sized for the
        maximum token, row and feature-entry counts, with every length zero.
    """
    # Instance `StepBuffer`
    step_buffer = _StepBuffer(
        x_actions=torch.zeros(num_buffers, buffer_size, 3, dtype=torch.long),
        x_ascensions=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
        x_rewards=torch.zeros(num_buffers, buffer_size, len(RewardStream), dtype=torch.float32),
        x_dones=torch.zeros(num_buffers, buffer_size, dtype=torch.float32),
        states=allocate_states(num_buffers, buffer_size),
    )

    # Workers and the Actor exchange these tensors across processes
    for x_tensor in iter_tensors(step_buffer):
        x_tensor.share_memory_()

    # Return one `StepBuffer` view per buffer
    return [map_tensors(step_buffer, lambda x_tensor: x_tensor[idx_buffer]) for idx_buffer in range(num_buffers)]


def step_workers(
    workers: Workers, x_actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[EpisodeResult]]:
    """Have every worker step its games and write the new states, and wait for every reply.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`.
    x_actions : torch.Tensor
        (E, 3) long. Each game's action type, state-local token offset and Monster target rank.
        The actions are copied to shared CPU storage before any worker is woken.

    Returns
    -------
    x_rewards : torch.Tensor
        (E, K) float32 on the CPU. The rewards, independent of the workers' reusable storage.
    x_dones : torch.Tensor
        (E,) float32 on the CPU. The done flags, independent of the workers' reusable storage.
    episode_results : list[EpisodeResult]
        The episodes finished on this step, in queue order across workers.
    """
    if not workers._handles:
        raise RuntimeError("The workers have been stopped")
    
    step_buffers = workers._step_buffers
    num_workers = len(step_buffers)
    envs_per_worker = len(step_buffers[0].x_actions)
    num_envs = num_workers * envs_per_worker
    if x_actions.shape != (num_envs, 3) or x_actions.dtype != torch.long:
        raise ValueError("Worker actions must have shape (num_envs, 3) and dtype torch.long")

    # One transfer off the device, then each worker's share of the CPU copy
    for step_buffer, x_actions_worker in zip(
        step_buffers, x_actions.cpu().chunk(len(step_buffers))
    ):
        step_buffer.x_actions.copy_(x_actions_worker)

    # Signal workers that their actions are ready for consumption
    for worker_handle in workers._handles:
        worker_handle.sph_request.release()

    # Wait for the workers to act on their environments
    _wait_for_replies(workers._handles)

    # Collect instant rewards and done flags
    x_rewards = torch.cat([step_buffer.x_rewards for step_buffer in step_buffers])
    x_dones = torch.cat([step_buffer.x_dones for step_buffer in step_buffers])

    # Get `EpisodeResult`s from finished workers
    episode_results = []
    for _ in range(int(x_dones.sum())):
        while True:
            try:
                episode_results.append(workers._episode_result_queue.get(timeout=1.0))
                break
            except Empty:
                _check_workers_alive(workers._handles)

    return x_rewards, x_dones, episode_results


def write_ascensions(workers: Workers, ascension_current: float) -> None:
    """Write each game's ascension under the soft-curriculum position.

    A game takes its new ascension the next time it resets.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`.
    ascension_current : float
        The soft-curriculum position.
    """
    step_buffers = workers._step_buffers
    num_envs = len(step_buffers) * len(step_buffers[0].x_ascensions)
    x_ascensions = torch.tensor(ascensions(ascension_current, num_envs), dtype=torch.long)
    for step_buffer, x_ascensions_worker in zip(
        step_buffers, x_ascensions.chunk(len(step_buffers))
    ):
        step_buffer.x_ascensions.copy_(x_ascensions_worker)


def _wait_for_replies(worker_handles: list[_WorkerHandle]) -> None:
    """Wait for every worker's reply, checking that it and the Learner are still alive.

    Parameters
    ----------
    worker_handles : list[_WorkerHandle]
        The worker processes and their reply semaphores.
    """
    for worker_handle in worker_handles:
        while not worker_handle.sph_reply.acquire(timeout=10.0):
            _check_workers_alive(worker_handles)


def _check_workers_alive(worker_handles: list[_WorkerHandle]) -> None:
    """Check that every worker and the Learner are still alive while waiting for a reply.

    Parameters
    ----------
    worker_handles : list[_WorkerHandle]
        The worker processes whose replies or episode records are pending.
    """
    for worker_idx, worker_handle in enumerate(worker_handles):
        if not worker_handle.process.is_alive():
            raise RuntimeError(
                f"Rollout worker {worker_idx} died (exitcode {worker_handle.process.exitcode})"
            )

    if not parent_alive():
        raise RuntimeError("The Learner died while waiting for rollout workers")


def snapshot_to_device(workers: Workers, device: torch.device) -> tuple[StateBatch, torch.Tensor]:
    """Copy every game's state out of the step buffers onto the device.

    The workers' rows are joined on the CPU and copied onto the device in one copy, with the
    state's counts kept on the CPU. The copy is the snapshot: the rollout keeps it while the
    workers overwrite the step on the next request.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`, after startup or a completed step.
    device : torch.device
        The device to copy onto.

    Returns
    -------
    state_batch : StateBatch
        Every game's state, on the device.
    x_counts_cpu : torch.Tensor
        (E, number_of_Jaggeds) long. Each of the E games' rows in each Jagged, on the CPU.
    """
    if not workers._handles:
        raise RuntimeError("The workers have been stopped")

    # Read `StateBatch`es from workers
    state_batches = [read_states(step_buffer.states) for step_buffer in workers._step_buffers]

    # Concatenate them into a single `StateBatch`
    state_batch = cat_states(state_batches)

    # Send the `StateBatch` to the device
    return to_device(state_batch, device), count_jagged_rows(state_batch)


def stop_workers(workers: Workers) -> None:
    """Set the stop flag, wake every worker to see it, and kill any still running after 5 s.

    Reap the processes and close the episode queue. Calling it again has no effect.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`; it cannot be stepped or snapshotted afterward.
    """
    workers._x_stop.fill_(True)
    for worker_handle in workers._handles:
        worker_handle.sph_request.release()

    for worker_handle in workers._handles:
        worker_handle.process.join(timeout=5)
        if worker_handle.process.is_alive():
            worker_handle.process.kill()
            worker_handle.process.join()

    workers._handles.clear()
    workers._episode_result_queue.close()


def _worker_loop(
    step_buffer: _StepBuffer,
    episode_result_queue: Queue[EpisodeResult],
    sph_request: Semaphore,
    sph_reply: Semaphore,
    x_stop: torch.Tensor,
    seed: int,
    gamma: float,
) -> None:
    """Run one shard of the games until the stop flag is set.

    It writes their first states, then on each request steps every game with its sampled action
    and writes the rewards, dones and new states. A game whose episode ended is replaced by a
    new one at its current ascension.

    Parameters
    ----------
    step_buffer : _StepBuffer
        This worker's shared memory.
    episode_result_queue : Queue[EpisodeResult]
        Where finished episodes go, shared by every worker.
    sph_request : Semaphore
        The step loop's request, released once per step.
    sph_reply : Semaphore
        The worker's reply, released once its games' states are written.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.
    seed : int
        The worker's seed, fixed for the run.
    gamma : float
        The discount the games' rewards and discounted returns use.
    """
    set_rng_seeds(seed)
    torch.set_num_threads(1)

    # Initialize environments and gather initial observations
    batch_env = []
    batch_game_state = []
    episode_results = []
    for ascension in step_buffer.x_ascensions.tolist():
        env, game_state = new_env(ascension, random.randint(0, 2**31 - 1))
        batch_env.append(env)
        batch_game_state.append(game_state)
        episode_results.append(EpisodeResult()) 

    # Write initial encodings into the shared step buffer and signal the Actor
    encode_states_into(
        step_buffer.states, batch_game_state, [env.get_legal_actions() for env in batch_env]
    )
    sph_reply.release()

    # Loop
    while _wait_for_request(sph_request, x_stop):
        batch_ascension = step_buffer.x_ascensions.tolist()

        for idx_env, (idx_action_kind, idx_token, idx_target) in enumerate(step_buffer.x_actions.tolist()):
            # Transform the model's numeric output into an engine Action
            action = engine_action(idx_action_kind, idx_token, idx_target, batch_game_state[idx_env])

            # Step the environment
            game_state_next, reward, done = step_env(
                batch_env[idx_env], batch_game_state[idx_env], episode_results[idx_env], action, gamma
            )

            # Write reward and done status into the shared step buffer 
            step_buffer.x_rewards[idx_env] = torch.from_numpy(reward)
            step_buffer.x_dones[idx_env] = float(done)

            # If done, put the episode's results into the queue and reset the environment
            if done:
                episode_result_queue.put(episode_results[idx_env])
                episode_results[idx_env] = EpisodeResult()
                batch_env[idx_env], game_state_next = new_env(
                    batch_ascension[idx_env], random.randint(0, 2**31 - 1)
                )

            # Overwrite current game state
            batch_game_state[idx_env] = game_state_next

        # Write the step's encoded game states into the shared step buffer and singal the Actor
        encode_states_into(
            step_buffer.states, batch_game_state, [env.get_legal_actions() for env in batch_env]
        )
        sph_reply.release()


def _wait_for_request(sph_request: Semaphore, x_stop: torch.Tensor) -> bool:
    """Wait for the step loop's next request.

    Parameters
    ----------
    sph_request : Semaphore
        The step loop's request, released once per step.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.

    Returns
    -------
    bool
        False once stopped, or if the Actor is gone; True otherwise.
    """
    while not sph_request.acquire(timeout=1.0):
        if not parent_alive():  # The Actor died without stopping this worker
            return False
    return not x_stop
