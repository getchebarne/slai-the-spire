from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Semaphore

import torch
import torch.multiprocessing as mp

from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.envs import Shard
from src.rl.algorithms.actor_critic.envs import ShardSettings
from src.rl.algorithms.actor_critic.envs import make_shard
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.encoding.state import empty_game_state
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.masks import engine_action
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
    """What the step loop and its workers share"""

    step: X_Step
    curriculum_c: torch.Tensor  # live curriculum position
    stop: torch.Tensor  # a lock-free flag: a worker killed holding an Event's lock would hang it
    episodes: Queue[EpisodeStats]


@dataclass
class Worker:
    """One worker process and its semaphores: the step loop's request, the worker's reply"""

    process: BaseProcess
    request: Semaphore
    reply: Semaphore


@dataclass
class RolloutConfig:
    num_envs: int
    rollout_workers: int
    rollout_length: int


def start_workers(
    rollout_config: RolloutConfig, curriculum: AscensionCurriculum, gamma: float, seed: int
) -> tuple[Exchange, list[Worker]]:
    """Start `rollout_workers` processes, each stepping an equal share of the `num_envs` games,
    and wait until every one has written its games' first states"""
    assert (
        rollout_config.num_envs % rollout_config.rollout_workers == 0
    ), "num_envs must divide evenly across workers"
    ctx = mp.get_context("spawn")  # CUDA is live in this process; fork is unsafe
    exchange = Exchange(
        step=X_Step(
            actions=torch.zeros(rollout_config.num_envs, 4, dtype=torch.long).share_memory_(),
            rewards=torch.zeros(rollout_config.num_envs, len(RewardStream)).share_memory_(),
            dones=torch.zeros(rollout_config.num_envs).share_memory_(),
            game_state=share_memory(empty_game_state(rollout_config.num_envs)),
            num_regions=rollout_config.rollout_workers,
        ),
        curriculum_c=torch.tensor(curriculum.ascension_start, dtype=torch.float64).share_memory_(),
        stop=torch.tensor(False).share_memory_(),
        episodes=ctx.Queue(),
    )
    per = rollout_config.num_envs // rollout_config.rollout_workers
    workers: list[Worker] = []
    try:
        for w in range(rollout_config.rollout_workers):
            settings = ShardSettings(
                num_envs=per,
                first_env=w * per,
                total_envs=rollout_config.num_envs,
                gamma=gamma,
                ascension_start=curriculum.ascension_start,
                ascension_max=curriculum.ascension_max,
                seed=seed * 7919 + w,
            )

            # Semaphores: an Event.set() blocks until its sleepers wake; a killed worker hangs it
            request, reply = ctx.Semaphore(0), ctx.Semaphore(0)
            process = ctx.Process(
                target=_worker_loop, args=(exchange, request, reply, settings), daemon=True
            )
            process.start()
            workers.append(Worker(process, request, reply))
        _wait_for_replies(workers)
    except BaseException:
        stop_workers(exchange, workers)
        raise
    return exchange, workers


def step_workers(workers: list[Worker]) -> None:
    """Have every worker step its games and write the new states, and wait for every reply"""
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
            if not parent_alive():  # a hung worker must not outlive the learner
                raise RuntimeError(f"the learner died while rollout worker {w} stepped")


def snapshot_to_device(step: X_Step, device: torch.device) -> tuple[X_GameState, torch.Tensor]:
    """Every game's state out of the shared step, the workers' rows joined on the CPU and copied
    onto the device in one copy, with its counts kept on the CPU. The copy is the snapshot: the
    rollout keeps it while the workers overwrite the step on the next request."""
    x = join_regions(step.game_state, step.num_regions)
    return to_device(x, device), count_blocks(x)


def drain_episodes(episodes: Queue[EpisodeStats]) -> list[EpisodeStats]:
    completed = []
    while not episodes.empty():
        completed.append(episodes.get())
    return completed


def stop_workers(exchange: Exchange, workers: list[Worker]) -> None:
    """Set the stop flag, wake every worker to see it, and kill any still running after 5 s"""
    exchange.stop.fill_(True)
    for worker in workers:
        worker.request.release()
    for worker in workers:
        worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.kill()


def _worker_loop(
    exchange: Exchange, request: Semaphore, reply: Semaphore, settings: ShardSettings
) -> None:
    """Run one shard of the games: write their first states, then on each request step every
    game with its sampled action and write the rewards, dones and new states, until the stop
    flag is set"""
    set_rng_seeds(settings.seed)
    torch.set_num_threads(1)  # workers are the parallelism; don't oversubscribe
    step = _worker_region(exchange.step, settings)
    shard = make_shard(settings)
    _write_states(step, shard)
    reply.release()
    while _wait_for_request(request, exchange.stop):
        shard.ascension_c = float(exchange.curriculum_c)  # promotions apply on reset

        for j, (at, token_kind, rank, target) in enumerate(step.actions.tolist()):
            action = engine_action(at, token_kind, rank, target, shard.obs[j])
            reward, finished = step_env(shard, j, action, settings)
            step.rewards[j] = torch.from_numpy(reward)
            step.dones[j] = float(finished is not None)
            if finished is not None:
                exchange.episodes.put(finished)
        _write_states(step, shard)
        reply.release()


def _wait_for_request(request: Semaphore, stop: torch.Tensor) -> bool:
    """Wait for the step loop's next request: False once stopped, or if the actor is gone"""
    while not request.acquire(timeout=1.0):
        if not parent_alive():  # the actor died without stopping this worker
            return False
    return not stop


def _worker_region(step: X_Step, settings: ShardSettings) -> X_Step:
    """The worker's region of the shared step, one region of its own: its games' actions, rewards
    and done flags, and their counts, per-state fields and rows of every block"""
    games = slice(settings.first_env, settings.first_env + settings.num_envs)
    return X_Step(
        actions=step.actions[games],
        rewards=step.rewards[games],
        dones=step.dones[games],
        game_state=region_of_states(step.game_state, settings.first_env, settings.num_envs),
        num_regions=1,
    )


def _write_states(step: X_Step, shard: Shard) -> None:
    """Encode the shard's games and their masks into its region of the step"""
    cpu = torch.device("cpu")
    legal = [env.get_legal_actions() for env in shard.envs]
    x = encode_batch_game_state(shard.obs, cpu)
    build_masks(shard.obs, legal, x)
    write_rows(step.game_state, x)
