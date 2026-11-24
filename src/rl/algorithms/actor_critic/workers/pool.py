import random
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Semaphore
from queue import Empty

import torch
import torch.multiprocessing as mp
from slai import Action
from slai import ActionType
from slai import EffectRelicGrantRandom
from slai import EffectRelicLose
from slai import EventOption
from slai import GameEnv
from slai import GameState
from slai import RelicName

from src.rl.actions import engine_action
from src.rl.actions import model_action
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.algorithms.actor_critic.envs import new_env
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.batches import allocate_states
from src.rl.batches import cat_states
from src.rl.batches import iter_tensors
from src.rl.batches import map_tensors
from src.rl.batches import read_states
from src.rl.batches import to_device
from src.rl.encoding import encode_states_into
from src.rl.reward import RewardStream
from src.rl.scripted import step_scripted
from src.rl.tokens import get_entities
from src.rl.tokens import relic_held
from src.rl.types import Live
from src.rl.types import Reserved
from src.rl.types import States
from src.rl.utils import parent_alive
from src.rl.utils import set_rng_seeds


@dataclass
class _StepBuffer:
    """One worker's shared memory: every tensor it and the Actor exchange each step.

    The token and sparse arrays are sized for the worst case, and the encoders write their
    leading rows, state after state. `read_states` exposes the populated collections the model
    consumes; the rows past them are stale. Rewriting a buffer invalidates earlier reads of it.
    """

    # One row per game, written by the Actor
    x_action: torch.Tensor     # (B, 3) long: action type, state-local selection offset, Monster target rank
    x_ascension: torch.Tensor  # (B,) long. Each game's ascension under the live position

    # One row per game, written by the worker
    x_reward: torch.Tensor        # (B, K)
    x_done: torch.Tensor          # (B,)
    x_num_discount: torch.Tensor  # (B,) long: k in the step's discount, γ^k
    x_forced: torch.Tensor        # (B,) bool: the worker wrote a forced action over x_action

    states_reserved: States[Reserved]  # The encoded states, at worst-case capacity


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
    num_envs: int,
    num_workers: int,
    ascension_current: float,
    gamma: float,
    seed: int,
    pandora_share: float,
    frozen_eye_share: float,
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
    pandora_share : float
        The probability that each new game starts from Pandora's Box.
    frozen_eye_share : float
        The probability that each new game plays Frozen Eye's Shop purchase whenever it's legal.

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

    # Create workers' contaienr and write the curriculum's current ascension level
    workers = Workers(step_buffers, [], ctx.Queue(), x_stop)
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
                    pandora_share,
                    frozen_eye_share,
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
        maximum token, row and feature-entry counts, with every count zero.
    """
    # Instance `StepBuffer`
    step_buffer = _StepBuffer(
        x_action=torch.zeros(num_buffers, buffer_size, 3, dtype=torch.long),
        x_ascension=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
        x_reward=torch.zeros(num_buffers, buffer_size, len(RewardStream), dtype=torch.float32),
        x_done=torch.zeros(num_buffers, buffer_size, dtype=torch.float32),
        x_num_discount=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
        x_forced=torch.zeros(num_buffers, buffer_size, dtype=torch.bool),
        states_reserved=allocate_states(num_buffers, buffer_size),
    )

    # Workers and the Actor exchange these tensors across processes
    for x_tensor in iter_tensors(step_buffer):
        x_tensor.share_memory_()

    # Return one `StepBuffer` view per buffer
    return [map_tensors(step_buffer, lambda x_tensor: x_tensor[idx_buffer]) for idx_buffer in range(num_buffers)]


def step_workers(
    workers: Workers, x_action: torch.Tensor
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[EpisodeResult]
]:
    """Have every worker step its games and write the new states, and wait for every reply.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`.
    x_action : torch.Tensor
        (E, 3) long. Each game's action type, state-local selection offset and Monster target rank.
        The actions are copied to shared CPU storage before any worker is woken.

    Returns
    -------
    x_action_played : torch.Tensor
        (E, 3) long on the CPU. The actions played: each game's row of `x_action`, or the forced
        action its worker wrote over it, independent of the workers' reusable storage.
    x_forced : torch.Tensor
        (E,) bool on the CPU. Whether each game's action was forced, independent of the workers'
        reusable storage.
    x_reward : torch.Tensor
        (E, K) float32 on the CPU. The rewards, independent of the workers' reusable storage.
    x_done : torch.Tensor
        (E,) float32 on the CPU. The done flags, independent of the workers' reusable storage.
    x_num_discount : torch.Tensor
        (E,) long on the CPU. Each step's k in γ^k, independent of the workers' reusable storage.
    episode_results : list[EpisodeResult]
        The episodes finished on this step, in queue order across workers.
    """
    if not workers._handles:
        raise RuntimeError("The workers have been stopped")
    
    step_buffers = workers._step_buffers
    num_workers = len(step_buffers)
    envs_per_worker = len(step_buffers[0].x_action)
    num_envs = num_workers * envs_per_worker
    if x_action.shape != (num_envs, 3) or x_action.dtype != torch.long:
        raise ValueError("Worker actions must have shape (num_envs, 3) and dtype torch.long")

    # One transfer off the device, then each worker's share of the CPU copy
    for step_buffer, x_action_worker in zip(
        step_buffers, x_action.cpu().chunk(len(step_buffers))
    ):
        step_buffer.x_action.copy_(x_action_worker)

    # Signal workers that their actions are ready for consumption
    for worker_handle in workers._handles:
        worker_handle.sph_request.release()

    # Wait for the workers to act on their environments
    _wait_for_replies(workers._handles)

    # Collect the actions played and whether each was forced
    x_action_played = torch.cat([step_buffer.x_action for step_buffer in step_buffers])
    x_forced = torch.cat([step_buffer.x_forced for step_buffer in step_buffers])

    # Collect instant rewards and done flags
    x_reward = torch.cat([step_buffer.x_reward for step_buffer in step_buffers])
    x_done = torch.cat([step_buffer.x_done for step_buffer in step_buffers])
    x_num_discount = torch.cat([step_buffer.x_num_discount for step_buffer in step_buffers])

    # Get `EpisodeResult`s from finished workers
    episode_results = []
    for _ in range(int(x_done.sum())):
        while True:
            try:
                episode_results.append(workers._episode_result_queue.get(timeout=1.0))
                break
            except Empty:
                _check_workers_alive(workers._handles)

    return x_action_played, x_forced, x_reward, x_done, x_num_discount, episode_results


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
    num_envs = len(step_buffers) * len(step_buffers[0].x_ascension)
    x_ascension = torch.tensor(ascensions(ascension_current, num_envs), dtype=torch.long)
    for step_buffer, x_ascension_worker in zip(
        step_buffers, x_ascension.chunk(len(step_buffers))
    ):
        step_buffer.x_ascension.copy_(x_ascension_worker)


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
    for idx_worker, worker_handle in enumerate(worker_handles):
        if not worker_handle.process.is_alive():
            raise RuntimeError(
                f"Rollout worker {idx_worker} died (exitcode {worker_handle.process.exitcode})"
            )

    if not parent_alive():
        raise RuntimeError("The Learner died while waiting for rollout workers")


def snapshot_to_device(workers: Workers, device: torch.device) -> States[Live]:
    """Copy every game's state out of the step buffers onto the device.

    The workers' rows are joined on the CPU and copied onto the device in one transfer. The
    snapshot owns its device rows and CPU totals; the rollout keeps both while the workers
    overwrite the step on the next request.

    Parameters
    ----------
    workers : Workers
        The handle returned by `start_workers`, after startup or a completed step.
    device : torch.device
        The device to copy onto.

    Returns
    -------
    States[Live]
        Every game's state, with device data and owned CPU totals.
    """
    if not workers._handles:
        raise RuntimeError("The workers have been stopped")

    # Read populated states from workers
    batch_states = [
        read_states(step_buffer.states_reserved) for step_buffer in workers._step_buffers
    ]

    # Concatenate them into a single `States`
    states = cat_states(batch_states)

    # Send the `States` to the device
    return to_device(states, device)


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
    pandora_share: float,
    frozen_eye_share: float,
) -> None:
    """Run one shard of the games until the stop flag is set.

    It writes their first states, then on each request steps every game with its sampled action,
    or the Frozen Eye purchase the game forces, and writes the forced actions and flags, the
    rewards, dones, discount counts and new states. A game whose episode ended is replaced by a
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
    pandora_share : float
        The probability that each new game starts from Pandora's Box.
    frozen_eye_share : float
        The probability that each new game plays Frozen Eye's Shop purchase whenever it's legal.
    """
    set_rng_seeds(seed)
    torch.set_num_threads(1)

    # Initialize environments and gather initial observations
    batch_env = []
    batch_game_state = []
    batch_frozen_eye_forced = []
    episode_results = []
    for ascension in step_buffer.x_ascension.tolist():
        env, game_state, frozen_eye_forced = _new_env_training(
            ascension, pandora_share, frozen_eye_share
        )
        batch_env.append(env)
        batch_game_state.append(game_state)
        batch_frozen_eye_forced.append(frozen_eye_forced)
        episode_results.append(EpisodeResult()) 

    # Write initial encodings into the shared step buffer and signal the Actor
    batch_legal_actions = [env.get_legal_actions() for env in batch_env]
    encode_states_into(
        step_buffer.states_reserved,
        batch_game_state,
        batch_legal_actions,
    )
    sph_reply.release()

    # Loop
    while _wait_for_request(sph_request, x_stop):
        batch_ascension = step_buffer.x_ascension.tolist()

        for idx_env, (idx_action_type, idx_selection, idx_target) in enumerate(step_buffer.x_action.tolist()):
            # Transform the model's numeric output into an engine Action
            action = engine_action(idx_action_type, idx_selection, idx_target, batch_game_state[idx_env])

            # Step the environment
            game_state_next, reward, num_discounts, done, action_played = step_env(
                batch_env[idx_env],
                batch_game_state[idx_env],
                batch_legal_actions[idx_env],
                episode_results[idx_env],
                action,
                gamma,
                batch_frozen_eye_forced[idx_env],
            )

            # Write a forced action over the Actor's row, in the model's format, and the flag
            forced = action_played != action
            if forced:
                step_buffer.x_action[idx_env] = torch.tensor(
                    model_action(action_played, batch_game_state[idx_env])
                )
            step_buffer.x_forced[idx_env] = forced

            # Write reward and done status into the shared step buffer 
            step_buffer.x_reward[idx_env] = torch.from_numpy(reward)
            step_buffer.x_done[idx_env] = float(done)
            step_buffer.x_num_discount[idx_env] = num_discounts

            # If done, put the episode's results into the queue and reset the environment
            if done:
                episode_result_queue.put(episode_results[idx_env])
                episode_results[idx_env] = EpisodeResult()
                batch_env[idx_env], game_state_next, batch_frozen_eye_forced[idx_env] = (
                    _new_env_training(batch_ascension[idx_env], pandora_share, frozen_eye_share)
                )

            # Overwrite current game state
            batch_game_state[idx_env] = game_state_next

        # Write the step's encoded game states into the shared step buffer and singal the Actor
        batch_legal_actions = [env.get_legal_actions() for env in batch_env]
        encode_states_into(
            step_buffer.states_reserved,
            batch_game_state,
            batch_legal_actions,
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


def _new_env_training(
    ascension: int, pandora_share: float, frozen_eye_share: float
) -> tuple[GameEnv, GameState, bool]:
    """Start a training game at `ascension` from a new random seed.

    With probability `pandora_share` the game starts from Pandora's Box, and with probability
    `frozen_eye_share` it forces Frozen Eye's Shop purchase whenever it's legal.

    Parameters
    ----------
    ascension : int
        The game's ascension.
    pandora_share : float
        The probability that the game starts from Pandora's Box.
    frozen_eye_share : float
        The probability that the game plays Frozen Eye's Shop purchase whenever it's legal.

    Returns
    -------
    env : GameEnv
        The game, with the run's `NEOW` setting.
    game_state : GameState
        Its first state.
    frozen_eye_forced : bool
        Whether the game plays Frozen Eye's Shop purchase whenever it's legal.
    """
    # A share of 0 skips this draw, so the seeds stay those of a run without forced purchases
    frozen_eye_forced = frozen_eye_share > 0 and random.random() < frozen_eye_share

    # A share of 0 skips this draw, so the seeds stay those of a run without Pandora starts
    if pandora_share > 0 and random.random() < pandora_share:
        env, game_state = _new_env_pandora(ascension)
    else:
        env, game_state = new_env(ascension, random.randint(0, 2**31 - 1))

    return env, game_state, frozen_eye_forced


def _new_env_pandora(ascension: int) -> tuple[GameEnv, GameState]:
    """Start a game at `ascension` from Neow's boss Relic swap onto Pandora's Box.

    Each try draws a new random seed and takes the swap outside the policy, until the swap
    grants Pandora's Box: about one seed in 22 does.

    Parameters
    ----------
    ascension : int
        The game's ascension.

    Returns
    -------
    env : GameEnv
        The game, with the run's `NEOW` setting.
    game_state : GameState
        The first state after the swap and the scripted steps it triggers, every starter
        Strike and Defend transformed.
    """
    while True:
        env, game_state = new_env(ascension, random.randint(0, 2**31 - 1))

        # Neow's boss Relic swap: the option that loses a Relic and grants a random one
        options = (
            []
            if game_state.event is None
            else get_entities(game_state, game_state.event.id_event_options, EventOption)
        )
        option_swap = next(
            option
            for option in options
            if {type(effect) for effect in option.effects}
            >= {EffectRelicLose, EffectRelicGrantRandom}
        )

        # Take the swap outside the policy, and keep the game if it granted Pandora's Box
        game_state, _, _ = step_scripted(
            env, Action(ActionType.EventOptionSelect, [option_swap.id])
        )
        if relic_held(game_state, RelicName.PandorasBox):
            return env, game_state
