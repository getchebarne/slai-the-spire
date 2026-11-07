import random
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Semaphore

import torch
import torch.multiprocessing as mp
from slai import GameEnv
from slai import GameState

from src.rl.actions import engine_action
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import new_env
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.algorithms.actor_critic.stats import EpisodeStats
from src.rl.encoding.state import encode_batch_game_state
from src.rl.packed import cat_states
from src.rl.packed import count_blocks
from src.rl.packed import to_device
from src.rl.reward import RewardStream
from src.rl.shared import _StateSlot
from src.rl.shared import allocate_slots
from src.rl.shared import read_slot
from src.rl.shared import write_slot
from src.rl.types import StateBatch
from src.rl.utils import parent_alive
from src.rl.utils import set_rng_seeds


@dataclass
class StepBuffer:
    """What the Actor and its workers exchange: one step of every game, their ascensions and
    finished episodes.

    The action the Actor sampled, then the reward, done flag and state, masks included, the
    worker wrote back. Each worker owns a state slot and a disjoint slice of the per-game
    tensors. The slot publishes the live shapes of its reusable storage with each reply.
    """

    x_actions: torch.Tensor  # (E, 3) long: action type, state-local token offset, Monster target rank
    x_rewards: torch.Tensor  # (E, K)
    x_dones: torch.Tensor  # (E,)
    slots: list[_StateSlot]  # One per worker, in game order
    x_ascensions: torch.Tensor  # (E,) long. Each game's ascension under the live position
    episodes: Queue[EpisodeStats]


@dataclass
class WorkerHandle:
    """One worker process, its semaphores (the step loop's request, its reply) and stop flag."""

    process: BaseProcess
    sph_request: Semaphore
    sph_reply: Semaphore
    x_stop: torch.Tensor  # A lock-free flag: a worker killed holding an Event's lock would hang it


def start_workers(
    num_envs: int, rollout_workers: int, ascension_current: float, gamma: float, seed: int
) -> tuple[StepBuffer, list[WorkerHandle]]:
    """Start `rollout_workers` processes, each stepping an equal share of the `num_envs` games.

    It then waits until every one has written its games' first states.

    Parameters
    ----------
    num_envs : int
        The number of games; they must divide evenly across the workers.
    rollout_workers : int
        The number of worker processes.
    ascension_current : float
        The soft-curriculum position the games start at.
    gamma : float
        The discount the games' rewards and discounted returns use.
    seed : int
        The seed each worker's own seed derives from.

    Returns
    -------
    step_buffer : StepBuffer
        What the Actor and its workers exchange.
    worker_handles : list[WorkerHandle]
        The worker processes and their semaphores.
    """
    if num_envs % rollout_workers != 0:
        raise RuntimeError("`num_envs` must divide evenly across workers")

    envs_per_worker = num_envs // rollout_workers
    ctx = mp.get_context("spawn")  # CUDA is live in this process; fork is unsafe
    step_buffer = StepBuffer(
        x_actions=torch.zeros(num_envs, 3, dtype=torch.long).share_memory_(),
        x_rewards=torch.zeros(num_envs, len(RewardStream)).share_memory_(),
        x_dones=torch.zeros(num_envs).share_memory_(),
        slots=allocate_slots(rollout_workers, envs_per_worker),
        x_ascensions=torch.tensor(
            ascensions(ascension_current, num_envs), dtype=torch.long
        ).share_memory_(),
        episodes=ctx.Queue(),
    )
    worker_handles = []
    try:
        for w in range(rollout_workers):
            games = range(w * envs_per_worker, (w + 1) * envs_per_worker)

            # Semaphores: an Event.set() blocks until its sleepers wake; a killed worker hangs it
            sph_request, sph_reply = ctx.Semaphore(0), ctx.Semaphore(0)
            x_stop = torch.tensor(False).share_memory_()
            process = ctx.Process(
                target=_worker_loop,
                args=(
                    _worker_region(step_buffer, games, w),
                    sph_request,
                    sph_reply,
                    x_stop,
                    seed * 7919 + w,
                    gamma,
                ),
                daemon=True,
            )
            process.start()
            worker_handles.append(WorkerHandle(process, sph_request, sph_reply, x_stop))

        # Wait for all Workers to reply
        _wait_for_replies(worker_handles)

    except BaseException:
        stop_workers(worker_handles)
        raise

    return step_buffer, worker_handles


def step_workers(worker_handles: list[WorkerHandle]) -> None:
    """Have every worker step its games and write the new states, and wait for every reply.

    Parameters
    ----------
    worker_handles : list[WorkerHandle]
        The worker processes and their semaphores.
    """
    for worker_handle in worker_handles:
        worker_handle.sph_request.release()

    _wait_for_replies(worker_handles)


def _wait_for_replies(worker_handles: list[WorkerHandle]) -> None:
    for worker_idx, worker_handle in enumerate(worker_handles):
        while not worker_handle.sph_reply.acquire(timeout=10.0):
            if not worker_handle.process.is_alive():
                raise RuntimeError(
                    f"Rollout worker {worker_idx} died (exitcode {worker_handle.process.exitcode})"
                )

            if not parent_alive():
                raise RuntimeError(f"The learner died while rollout worker {worker_idx} stepped")


def snapshot_to_device(
    step_buffer: StepBuffer, device: torch.device
) -> tuple[StateBatch, torch.Tensor]:
    """Copy every game's state out of the step buffer onto the device.

    The workers' rows are joined on the CPU and copied onto the device in one copy, with the
    state's counts kept on the CPU. The copy is the snapshot: the rollout keeps it while the
    workers overwrite the step on the next request.

    Parameters
    ----------
    step_buffer : StepBuffer
        The step buffer, as the workers wrote it.
    device : torch.device
        The device to copy onto.

    Returns
    -------
    x_game_state : StateBatch
        Every game's state, on the device.
    x_count_cpu : torch.Tensor
        (E, number of blocks) long. Each of the E games' rows in each block, on the CPU.
    """
    x_game_state = cat_states([read_slot(slot) for slot in step_buffer.slots])
    return to_device(x_game_state, device), count_blocks(x_game_state)


def drain_episodes(queue: Queue[EpisodeStats]) -> list[EpisodeStats]:
    episodes = []
    while not queue.empty():
        episodes.append(queue.get())
    return episodes


def stop_workers(worker_handles: list[WorkerHandle]) -> None:
    """Set every worker's stop flag, wake it to see it, and kill any still running after 5 s.

    Parameters
    ----------
    worker_handles : list[WorkerHandle]
        The worker processes, their semaphores and stop flags.
    """
    for worker_handle in worker_handles:
        worker_handle.x_stop.fill_(True)
        worker_handle.sph_request.release()

    for worker_handle in worker_handles:
        worker_handle.process.join(timeout=5)
        if worker_handle.process.is_alive():
            worker_handle.process.kill()


def _worker_loop(
    step_buffer_region: StepBuffer,
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
    step_buffer_region : StepBuffer
        This worker's state slot and its slices of the Actor's per-game tensors.
    sph_request : Semaphore
        The step loop's request, released once per step.
    sph_reply : Semaphore
        The worker's reply, released once its games' states are written.
    x_stop : torch.Tensor
        () bool. Its stop flag, in shared memory.
    seed : int
        The worker's seed, fixed for the run.
    gamma : float
        The discount the games' rewards and discounted returns use.
    """
    set_rng_seeds(seed)
    torch.set_num_threads(1)  # Workers are the parallelism; don't oversubscribe

    # Each game's tally lives in `batch_stats`, not per rollout, so an episode that spans
    # rollouts reports its true totals
    batch_env, batch_game_state = [], []
    for ascension in step_buffer_region.x_ascensions.tolist():
        env, game_state = new_env(ascension, random.randint(0, 2**31 - 1))
        batch_env.append(env)
        batch_game_state.append(game_state)
    batch_stats = [EpisodeStats() for _ in batch_env]
    _write_states(step_buffer_region, batch_env, batch_game_state)
    sph_reply.release()
    while _wait_for_request(sph_request, x_stop):
        batch_ascension = step_buffer_region.x_ascensions.tolist()  # Promotions apply on reset

        for j, (action_type, token_offset, target) in enumerate(
            step_buffer_region.x_actions.tolist()
        ):
            action = engine_action(action_type, token_offset, target, batch_game_state[j])
            game_state_next, reward, done = step_env(
                batch_env[j], batch_game_state[j], batch_stats[j], action, gamma
            )
            step_buffer_region.x_rewards[j] = torch.from_numpy(reward)
            step_buffer_region.x_dones[j] = float(done)
            if done:
                step_buffer_region.episodes.put(batch_stats[j])
                batch_stats[j] = EpisodeStats()
                batch_env[j], game_state_next = new_env(
                    batch_ascension[j], random.randint(0, 2**31 - 1)
                )
            batch_game_state[j] = game_state_next
        _write_states(step_buffer_region, batch_env, batch_game_state)
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


def _worker_region(step_buffer: StepBuffer, games: range, worker_index: int) -> StepBuffer:
    """Return the worker's region of the step buffer, one region of its own.

    The region holds its games' actions, rewards, done flags and ascensions, their counts,
    per-state fields and rows of every block, and the shared queue of finished episodes.

    Parameters
    ----------
    step_buffer : StepBuffer
        The step buffer.
    games : range
        The worker's games, by their indices among the run's games.
    worker_index : int
        The worker's state slot.

    Returns
    -------
    StepBuffer
        The worker's region, views of the step buffer's tensors.
    """
    return StepBuffer(
        x_actions=step_buffer.x_actions[games.start : games.stop],
        x_rewards=step_buffer.x_rewards[games.start : games.stop],
        x_dones=step_buffer.x_dones[games.start : games.stop],
        slots=[step_buffer.slots[worker_index]],
        x_ascensions=step_buffer.x_ascensions[games.start : games.stop],
        episodes=step_buffer.episodes,
    )


def _write_states(
    step_buffer_region: StepBuffer, batch_env: list[GameEnv], batch_game_state: list[GameState]
) -> None:
    """Encode the worker's games and their masks into its region of the step buffer.

    Parameters
    ----------
    step_buffer_region : StepBuffer
        The worker's region of the step buffer.
    batch_env : list[GameEnv]
        The worker's games.
    batch_game_state : list[GameState]
        Their states.
    """
    cpu = torch.device("cpu")
    batch_legal_actions = [env.get_legal_actions() for env in batch_env]
    x_game_state = encode_batch_game_state(batch_game_state, batch_legal_actions, cpu)
    write_slot(step_buffer_region.slots[0], x_game_state)
