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
from src.rl.masks import empty_masks
from src.rl.masks import engine_action
from src.rl.reward import RewardStream
from src.rl.types import X_EntityMask
from src.rl.types import X_EntityPairs
from src.rl.types import X_GameState
from src.rl.types import X_Mask
from src.rl.types import X_Owned
from src.rl.types import X_Rows
from src.rl.utils import set_rng_seeds


# The rollout's CPU work (engine step, reward, encode, masks) runs in worker processes, each
# over its shard of the games, while the step loop runs the forward and the sampling. Both
# sides share one X_Rows of torch shared tensors, so actions, rewards, states and masks move
# without pickling; every block sits at its cap


@dataclass
class Exchange:
    """What the step loop and its workers share"""

    rows: X_Rows
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
        rows=X_Rows(
            actions=torch.zeros(rollout_config.num_envs, 4, dtype=torch.long),
            rewards=torch.zeros(rollout_config.num_envs, len(RewardStream)),
            dones=torch.zeros(rollout_config.num_envs),
            game_state=empty_game_state(rollout_config.num_envs),
            mask_batch=empty_masks(rollout_config.num_envs),
            batch_size=[rollout_config.num_envs],
        ).share_memory_(),
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


def snapshot_to_device(rows: X_Rows, device: torch.device) -> tuple[X_GameState, X_Mask]:
    """Copy every game's row out of the shared rows onto the device as an X_GameState/X_Mask
    pair, each block trimmed from its cap to the rows' widest valid prefix, and its pairs to the
    most a row uses. The copy is the snapshot: the rollout keeps it while the workers overwrite
    the rows on the next request."""

    # Shallow copies: a cut block replaces its entry here, not in the shared rows
    x, mb = rows.game_state.copy(), rows.mask_batch.copy()

    # The six entity blocks, each cut with its masks, then the owned families
    x.cards, mb.cards = _cut_entity(x.cards, mb.cards)
    x.relics, mb.relics = _cut_entity(x.relics, mb.relics)
    x.potions, mb.potions = _cut_entity(x.potions, mb.potions)
    x.monsters, mb.monsters = _cut_entity(x.monsters, mb.monsters)
    mb.monster_is_target = mb.monster_is_target[:, : x.monsters.x_slice_kind.shape[1]]
    x.event_options, mb.event_options = _cut_entity(x.event_options, mb.event_options)
    x.character, mb.character = _cut_entity(x.character, mb.character)
    x.health_block = _cut(x.health_block)
    x.prices = _cut(x.prices)
    x.effects = _cut(x.effects)
    x.modifiers = _cut(x.modifiers)

    # Every field packed into one pinned buffer and sent in one copy: a copy per field from
    # the shared (pageable) memory makes the CPU wait for the GPU once per field
    fields: list[torch.Tensor] = []
    x.apply(fields.append)
    mb.apply(fields.append)
    starts, end = [], 0
    for t in fields:
        starts.append(end)
        end += -(-t.numel() * t.element_size() // 8) * 8  # 8-byte aligned for every dtype
    staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for t, start in zip(fields, starts):
        _bytes_as(staging, start, t).copy_(t)
    packed = staging.to(device, non_blocking=True) if device.type == "cuda" else staging
    pieces = iter(_bytes_as(packed, start, t) for t, start in zip(fields, starts))
    return x.apply(lambda t: next(pieces)), mb.apply(lambda t: next(pieces))


def _cut[B: (X_EntityPairs, X_Owned)](block: B) -> B:
    """`block` cut to the rows' widest valid prefix, and its pairs to the most a row uses"""
    # The widest valid prefix of a kind/owner row across the rows.
    w = int(block.mask.sum(1).max())

    # The pairs any row uses, at least one: a row's nonzero values come first
    k = max(int(block.x_value[:, :w].numpy().any(axis=(0, 1)).sum()), 1)
    return block.apply(lambda t: t[:, :w, :k] if t.dim() == 3 else t[:, :w])


def _cut_entity(block: X_EntityPairs, masks: X_EntityMask) -> tuple[X_EntityPairs, X_EntityMask]:
    """An entity block cut as _cut cuts it, and its masks cut to the same tokens"""
    block_cut = _cut(block)
    w = block_cut.x_slice_kind.shape[1]
    return block_cut, masks.apply(lambda t: t[..., :w])


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    """The bytes of `buffer` from `start` viewed as a tensor of `like`'s dtype and shape"""
    n = like.numel() * like.element_size()
    return buffer[start : start + n].view(like.dtype).view(like.shape)


def drain_episodes(episodes: Queue[EpisodeStats]) -> list[EpisodeStats]:
    completed = []
    while not episodes.empty():
        completed.append(episodes.get())
    return completed


def parent_alive() -> bool:
    """Whether the process that started this one is still alive"""
    parent = mp.parent_process()
    return parent is not None and parent.is_alive()


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
    rows = exchange.rows[settings.first_env : settings.first_env + settings.num_envs]
    pad_x, pad_mb = empty_game_state(settings.num_envs), empty_masks(settings.num_envs)
    shard = make_shard(settings)
    _write_states(rows, shard, pad_x, pad_mb)
    reply.release()
    while _wait_for_request(request, exchange.stop):
        shard.ascension_c = float(exchange.curriculum_c)  # promotions apply on reset

        for j, (at, kind, rank, target) in enumerate(rows.actions.tolist()):
            action = engine_action(at, kind, rank, target, shard.obs[j])
            reward, finished = step_env(shard, j, action, settings)
            rows.rewards[j] = torch.from_numpy(reward)
            rows.dones[j] = float(finished is not None)
            if finished is not None:
                exchange.episodes.put(finished)
        _write_states(rows, shard, pad_x, pad_mb)
        reply.release()


def _wait_for_request(request: Semaphore, stop: torch.Tensor) -> bool:
    """Wait for the step loop's next request: False once stopped, or if the actor is gone"""
    while not request.acquire(timeout=1.0):
        if not parent_alive():  # the actor died without stopping this worker
            return False
    return not stop


def _write_states(rows: X_Rows, shard: Shard, pad_x: X_GameState, pad_mb: X_Mask) -> None:
    """Encode the shard's games and their masks into its rows"""
    cpu = torch.device("cpu")
    legal = [env.get_legal_actions() for env in shard.envs]
    x = encode_batch_game_state(shard.obs, cpu)
    mb = build_masks(shard.obs, legal, x, cpu)

    # Every row back to padding, then every field of the encode into its leading corner
    rows.game_state.copy_(pad_x)
    rows.game_state.apply(_copy_prefix, x)
    rows.mask_batch.copy_(pad_mb)
    rows.mask_batch.apply(_copy_prefix, mb)


def _copy_prefix(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy an encode's field into the leading corner of its cap-wide shared field"""
    assert dst.dtype == src.dtype
    dst[tuple(slice(0, n) for n in src.shape)].copy_(src)
