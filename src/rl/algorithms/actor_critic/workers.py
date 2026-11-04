from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Semaphore

import torch
import torch.multiprocessing as mp

from src.rl.actions import engine_action
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import Shard
from src.rl.algorithms.actor_critic.envs import ShardSettings
from src.rl.algorithms.actor_critic.envs import make_shard
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.encoding.state import empty_game_state
from src.rl.encoding.state import encode_batch_game_state
from src.rl.packed import count_blocks
from src.rl.packed import join_regions
from src.rl.packed import region_of_states
from src.rl.packed import share_memory
from src.rl.packed import to_device
from src.rl.packed import write_rows
from src.rl.reward import RewardStream
from src.rl.types import X_GameState
from src.rl.types import X_Step
from src.rl.utils import parent_alive
from src.rl.utils import set_rng_seeds


# The rollout's CPU work (engine step, reward, encode, masks) runs in worker processes, each
# over its shard of the games, while the step loop runs the forward and the sampling. Both
# sides share one X_Step of torch shared tensors, so actions, rewards, states and masks move
# without pickling; each worker owns a region of it, sized for its games at every cap


@dataclass
class Exchange:
    """What the step loop and its workers share."""

    x_step: X_Step
    x_ascensions: torch.Tensor  # (E,) long. Each game's ascension under the live position
    x_stop: torch.Tensor  # A lock-free flag: a worker killed holding an Event's lock would hang it
    episodes: Queue[EpisodeStats]


@dataclass
class Worker:
    """One worker process and its semaphores: the step loop's request, the worker's reply."""

    process: BaseProcess
    request: Semaphore
    reply: Semaphore


@dataclass
class RolloutConfig:
    num_envs: int
    rollout_workers: int
    rollout_length: int


def start_workers(
    rollout_config: RolloutConfig, ascension_current: float, gamma: float, seed: int
) -> tuple[Exchange, list[Worker]]:
    """Start `rollout_workers` processes, each stepping an equal share of the `num_envs` games.

    It then waits until every one has written its games' first states.

    Parameters
    ----------
    rollout_config : RolloutConfig
        The number of games and of workers; the games must divide evenly across the workers.
    ascension_current : float
        The soft-curriculum position the games start at.
    gamma : float
        The discount the games' rewards and discounted returns use.
    seed : int
        The seed each worker's own seed derives from.

    Returns
    -------
    exchange : Exchange
        What the step loop and its workers share.
    workers : list[Worker]
        The worker processes and their semaphores.
    """
    assert (
        rollout_config.num_envs % rollout_config.rollout_workers == 0
    ), "num_envs must divide evenly across workers"
    ctx = mp.get_context("spawn")  # CUDA is live in this process; fork is unsafe
    exchange = Exchange(
        x_step=X_Step(
            x_actions=torch.zeros(rollout_config.num_envs, 4, dtype=torch.long).share_memory_(),
            x_rewards=torch.zeros(rollout_config.num_envs, len(RewardStream)).share_memory_(),
            x_dones=torch.zeros(rollout_config.num_envs).share_memory_(),
            x_gs=share_memory(empty_game_state(rollout_config.num_envs)),
            num_regions=rollout_config.rollout_workers,
        ),
        x_ascensions=torch.tensor(
            ascensions(ascension_current, rollout_config.num_envs), dtype=torch.long
        ).share_memory_(),
        x_stop=torch.tensor(False).share_memory_(),
        episodes=ctx.Queue(),
    )
    per = rollout_config.num_envs // rollout_config.rollout_workers
    workers: list[Worker] = []
    try:
        for w in range(rollout_config.rollout_workers):
            settings = ShardSettings(games=range(w * per, (w + 1) * per), seed=seed * 7919 + w)

            # Semaphores: an Event.set() blocks until its sleepers wake; a killed worker hangs it
            request, reply = ctx.Semaphore(0), ctx.Semaphore(0)
            process = ctx.Process(
                target=_worker_loop, args=(exchange, request, reply, settings, gamma), daemon=True
            )
            process.start()
            workers.append(Worker(process, request, reply))
        _wait_for_replies(workers)
    except BaseException:
        stop_workers(exchange, workers)
        raise
    return exchange, workers


def step_workers(workers: list[Worker]) -> None:
    """Have every worker step its games and write the new states, and wait for every reply.

    Parameters
    ----------
    workers : list[Worker]
        The worker processes and their semaphores.
    """
    for worker in workers:
        worker.request.release()
    _wait_for_replies(workers)


def _wait_for_replies(workers: list[Worker]) -> None:
    for w, worker in enumerate(workers):
        # Bounded wait + liveness check: a crashed worker must surface as an
        # error here, never as a silent hang on a dead semaphore.
        while not worker.reply.acquire(timeout=10.0):
            if not worker.process.is_alive():
                raise RuntimeError(
                    f"rollout worker {w} died (exitcode {worker.process.exitcode}); "
                    "see its traceback above"
                )
            if not parent_alive():  # A hung worker must not outlive the learner
                raise RuntimeError(f"the learner died while rollout worker {w} stepped")


def snapshot_to_device(x_step: X_Step, device: torch.device) -> tuple[X_GameState, torch.Tensor]:
    """Copy every game's state out of the shared step onto the device.

    The workers' rows are joined on the CPU and copied onto the device in one copy, with the
    state's counts kept on the CPU. The copy is the snapshot: the rollout keeps it while the
    workers overwrite the step on the next request.

    Parameters
    ----------
    x_step : X_Step
        The shared step, as the workers wrote it.
    device : torch.device
        The device to copy onto.

    Returns
    -------
    x_gs : X_GameState
        Every game's state, on the device.
    x_count_cpu : torch.Tensor
        (E, number of blocks) long. Each of the E games' rows in each block, on the CPU.
    """
    x_gs = join_regions(x_step.x_gs, x_step.num_regions)
    return to_device(x_gs, device), count_blocks(x_gs)


def drain_episodes(queue: Queue[EpisodeStats]) -> list[EpisodeStats]:
    episodes = []
    while not queue.empty():
        episodes.append(queue.get())
    return episodes


def stop_workers(exchange: Exchange, workers: list[Worker]) -> None:
    """Set the stop flag, wake every worker to see it, and kill any still running after 5 s.

    Parameters
    ----------
    exchange : Exchange
        What the step loop and its workers share.
    workers : list[Worker]
        The worker processes and their semaphores.
    """
    exchange.x_stop.fill_(True)
    for worker in workers:
        worker.request.release()
    for worker in workers:
        worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.kill()


def _worker_loop(
    exchange: Exchange, request: Semaphore, reply: Semaphore, settings: ShardSettings, gamma: float
) -> None:
    """Run one shard of the games until the stop flag is set.

    It writes their first states, then on each request steps every game with its sampled action
    and writes the rewards, dones and new states.

    Parameters
    ----------
    exchange : Exchange
        What the step loop and its workers share.
    request : Semaphore
        The step loop's request, released once per step.
    reply : Semaphore
        The worker's reply, released once its games' states are written.
    settings : ShardSettings
        The shard's settings, fixed for the run.
    gamma : float
        The discount the games' rewards and discounted returns use.
    """
    set_rng_seeds(settings.seed)
    torch.set_num_threads(1)  # Workers are the parallelism; don't oversubscribe
    x_step = _worker_region(exchange.x_step, settings)
    x_ascensions = exchange.x_ascensions[settings.games.start : settings.games.stop]
    shard = make_shard(x_ascensions.tolist())
    _write_states(x_step, shard)
    reply.release()
    while _wait_for_request(request, exchange.x_stop):
        batch_ascension = x_ascensions.tolist()  # Promotions apply on reset

        for j, (action_type, token_kind, rank, target) in enumerate(x_step.x_actions.tolist()):
            action = engine_action(
                action_type, token_kind, rank, target, shard.batch_game_state[j]
            )
            reward, episode = step_env(shard, j, action, gamma, batch_ascension[j])
            x_step.x_rewards[j] = torch.from_numpy(reward)
            x_step.x_dones[j] = float(episode is not None)
            if episode is not None:
                exchange.episodes.put(episode)
        _write_states(x_step, shard)
        reply.release()


def _wait_for_request(request: Semaphore, x_stop: torch.Tensor) -> bool:
    """Wait for the step loop's next request.

    Parameters
    ----------
    request : Semaphore
        The step loop's request, released once per step.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.

    Returns
    -------
    bool
        False once stopped, or if the actor is gone; True otherwise.
    """
    while not request.acquire(timeout=1.0):
        if not parent_alive():  # The actor died without stopping this worker
            return False
    return not x_stop


def _worker_region(x_step: X_Step, settings: ShardSettings) -> X_Step:
    """Return the worker's region of the shared step, one region of its own.

    The region holds its games' actions, rewards and done flags, and their counts, per-state
    fields and rows of every block.

    Parameters
    ----------
    x_step : X_Step
        The shared step.
    settings : ShardSettings
        The shard's settings, which name the worker's games.

    Returns
    -------
    X_Step
        The worker's region, views of the shared step's tensors.
    """
    games = slice(settings.games.start, settings.games.stop)
    return X_Step(
        x_actions=x_step.x_actions[games],
        x_rewards=x_step.x_rewards[games],
        x_dones=x_step.x_dones[games],
        x_gs=region_of_states(x_step.x_gs, settings.games.start, len(settings.games)),
        num_regions=1,
    )


def _write_states(x_step: X_Step, shard: Shard) -> None:
    """Encode the shard's games and their masks into its region of the step.

    Parameters
    ----------
    x_step : X_Step
        The worker's region of the shared step.
    shard : Shard
        The worker's games.
    """
    cpu = torch.device("cpu")
    batch_legal_actions = [env.get_legal_actions() for env in shard.batch_env]
    x_gs = encode_batch_game_state(shard.batch_game_state, batch_legal_actions, cpu)
    write_rows(x_step.x_gs, x_gs)
