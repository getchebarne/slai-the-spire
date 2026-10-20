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
from src.rl.reward import RewardStream
from src.rl.types import X_EffectPending
from src.rl.types import X_EntitySparse
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Step
from src.rl.utils import set_rng_seeds


# The rollout's CPU work (engine step, reward, encode, masks) runs in worker processes, each
# over its shard of the games, while the step loop runs the forward and the sampling. Both
# sides share one X_Step of torch shared tensors, so actions, rewards, states and masks move
# without pickling; every block sits at its cap


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
            actions=torch.zeros(rollout_config.num_envs, 4, dtype=torch.long),
            rewards=torch.zeros(rollout_config.num_envs, len(RewardStream)),
            dones=torch.zeros(rollout_config.num_envs),
            game_state=empty_game_state(rollout_config.num_envs),
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


def snapshot_to_device(step: X_Step, device: torch.device) -> X_GameState:
    """Copy every game's row out of the shared step onto the device as an X_GameState, each block
    trimmed from its cap to the rows' widest valid prefix, its masks with it, and its entries to
    the most a row uses. The copy is the snapshot: the rollout keeps it while the workers
    overwrite the step on the next request."""

    # A shallow copy: a cut block replaces its entry here, not in the shared step
    x = step.game_state.copy()

    # The six entity blocks, each cut with its masks, then the owned families and pending effect
    x.cards = _cut_entity(x.cards)
    x.relics = _cut_entity(x.relics)
    x.potions = _cut_entity(x.potions)
    x.monsters = _cut_entity(x.monsters)
    x.event_options = _cut_entity(x.event_options)
    x.character = _cut_entity(x.character)
    x.health_block = _cut_owned(x.health_block)
    x.prices = _cut_owned(x.prices)
    x.effects = _cut_owned(x.effects)
    x.modifiers = _cut_owned(x.modifiers)
    x.effect_pending = _cut_effect_pending(x.effect_pending)

    # Every field packed into one pinned buffer and sent in one copy: a copy per field from
    # the shared (pageable) memory makes the CPU wait for the GPU once per field
    fields: list[torch.Tensor] = []
    x.apply(fields.append)
    starts, end = [], 0
    for t in fields:
        starts.append(end)
        end += -(-t.numel() * t.element_size() // 8) * 8  # 8-byte aligned for every dtype
    staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for t, start in zip(fields, starts):
        _bytes_as(staging, start, t).copy_(t)
    packed = staging.to(device, non_blocking=True) if device.type == "cuda" else staging
    pieces = iter(_bytes_as(packed, start, t) for t, start in zip(fields, starts))
    return x.apply(lambda t: next(pieces))


def _cut_entity(block: X_EntitySparse) -> X_EntitySparse:
    """`block` cut to the rows' widest valid prefix, its entries to the most a row uses, and its
    masks to its tokens; field by field, as `x_legal`'s last dim is the action types"""
    w, k = _cut_sizes(block)
    return X_EntitySparse(
        x_column=block.x_column[:, :w, :k],
        x_value=block.x_value[:, :w, :k],
        x_slice_kind=block.x_slice_kind[:, :w],
        x_legal=block.x_legal[:, :w],
        x_needs_target=block.x_needs_target[:, :w],
        batch_size=block.batch_size,
    )


def _cut_owned(block: X_OwnedSparse) -> X_OwnedSparse:
    """`block` cut to the rows' widest valid prefix, and its entries to the most a row uses"""
    w, k = _cut_sizes(block)
    return X_OwnedSparse(
        x_column=block.x_column[:, :w, :k],
        x_value=block.x_value[:, :w, :k],
        x_owner_slice_kind=block.x_owner_slice_kind[:, :w],
        x_owner_rank=block.x_owner_rank[:, :w],
        batch_size=block.batch_size,
    )


def _cut_sizes(block: X_EntitySparse | X_OwnedSparse) -> tuple[int, int]:
    """The rows' widest valid prefix and the most entries a row uses: the sizes `block` is cut
    to"""
    # The widest valid prefix of a slice kind/owner row across the rows.
    w = int(block.mask.sum(1).max())

    # The entries any row uses, at least one: a row's nonzero values come first
    k = max(int(block.x_value[:, :w].numpy().any(axis=(0, 1)).sum()), 1)
    return w, k


def _cut_effect_pending(block: X_EffectPending) -> X_EffectPending:
    """`block` cut to the most entries a row uses"""
    # The entries any row uses, at least one: a row's nonzero values come first
    k = max(int(block.x_value.numpy().any(axis=0).sum()), 1)
    return block.apply(lambda t: t[:, :k])


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
    step = exchange.step[settings.first_env : settings.first_env + settings.num_envs]
    pad_x = empty_game_state(settings.num_envs)
    shard = make_shard(settings)
    _write_states(step, shard, pad_x)
    reply.release()
    while _wait_for_request(request, exchange.stop):
        shard.ascension_c = float(exchange.curriculum_c)  # promotions apply on reset

        for j, (at, slice_kind, rank, target) in enumerate(step.actions.tolist()):
            action = engine_action(at, slice_kind, rank, target, shard.obs[j])
            reward, finished = step_env(shard, j, action, settings)
            step.rewards[j] = torch.from_numpy(reward)
            step.dones[j] = float(finished is not None)
            if finished is not None:
                exchange.episodes.put(finished)
        _write_states(step, shard, pad_x)
        reply.release()


def _wait_for_request(request: Semaphore, stop: torch.Tensor) -> bool:
    """Wait for the step loop's next request: False once stopped, or if the actor is gone"""
    while not request.acquire(timeout=1.0):
        if not parent_alive():  # the actor died without stopping this worker
            return False
    return not stop


def _write_states(step: X_Step, shard: Shard, pad_x: X_GameState) -> None:
    """Encode the shard's games and their masks into its rows of the step"""
    cpu = torch.device("cpu")
    legal = [env.get_legal_actions() for env in shard.envs]
    x = encode_batch_game_state(shard.obs, cpu)
    build_masks(shard.obs, legal, x)

    # Every row back to padding, then every field of the encode into its leading corner
    step.game_state.copy_(pad_x)
    step.game_state.apply(_copy_prefix, x)


def _copy_prefix(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy an encode's field into the leading corner of its cap-wide shared field"""
    assert dst.dtype == src.dtype
    dst[tuple(slice(0, n) for n in src.shape)].copy_(src)
