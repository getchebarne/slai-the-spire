import math
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from types import FrameType

import numpy as np
import slai
import torch
import torch.multiprocessing  # noqa: F401 — registers tensor reductions (shm handles over pipes)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic.episode import PREFIX_TRAIN
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import aggregate_episodes
from src.rl.algorithms.actor_critic.evals import EVAL_ACT2_RATE
from src.rl.algorithms.actor_critic.evals import EVAL_FLOOR_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_REWARD_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_WIN_RATE
from src.rl.algorithms.actor_critic.evals import eval_battery_worker
from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.models import ActorCritic
from src.rl.reward import REWARD_STREAMS
from src.rl.reward import compute_reward
from src.rl.reward import cumulative_floor
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import Level
from src.rl.types import RolloutBuffer
from src.rl.types import TGameState
from src.rl.types import TMask
from src.rl.types import TPadded
from src.rl.utils import action_from_actiontype
from src.rl.utils import engine_action
from src.rl.utils import init_optimizer
from src.rl.utils import load_config
from src.rl.utils import shuffle_rollout_buffer


_AT_REST = ACTION_TYPE_BY_INT.index(slai.ActionType.Rest)
_AT_UPGRADE = ACTION_TYPE_BY_INT.index(slai.ActionType.CardUpgrade)


def _env_level(c: float, env_idx: int, num_envs: int, max_level: int) -> int:
    """Env -> ascension under soft-curriculum position c: the first
    round(frac(c)*num_envs) envs run the frontier level ceil(c), the rest floor(c)."""
    base = math.floor(c)
    n_frontier = round((c - base) * num_envs)
    return min(base + 1 if env_idx < n_frontier else base, max_level)


def _eval_level(c: float, max_level: int) -> int:
    """Frontier level the eval battery gates on: ceil(c) (== c at integers)."""
    return min(math.ceil(c), max_level)


def _curriculum_step(
    c: float,
    streak: int,
    eval_win_rate: float,
    eval_level: int,
    promote_win_rate: float,
    promote_streak: int,
    step: float,
    max_level: int,
) -> tuple[float, int]:
    """Advance the soft-curriculum position on sustained frontier mastery. Batteries
    from a stale level are skipped entirely (neither count nor reset the streak); a
    below-threshold frontier result resets it; `promote_streak` consecutive passes
    graduate another `step` fraction of envs. c is capped at max_level."""
    if eval_level != _eval_level(c, max_level):
        return c, streak
    if eval_win_rate < promote_win_rate:
        return c, 0
    streak += 1
    if streak < promote_streak:
        return c, streak
    return min(c + step, float(max_level)), 0


@dataclass
class EnvPool:
    envs: list[slai.GameEnv]
    obs: list[slai.GameState]
    ep_rewards: list[np.ndarray]  # (K,) per env, accumulated across the episode
    ep_lengths: list[int]
    completed: list[EpisodeStats]
    gamma: float
    ascension_c: float  # soft-curriculum position; envs adopt their level on reset
    ascension_max: int
    # Sharded pools cover env range [idx_offset, idx_offset+len(envs)) of a
    # total_envs-wide fleet; the curriculum level is a function of the GLOBAL
    # env index so the level distribution is shard-invariant.
    idx_offset: int = 0
    total_envs: int = 0  # 0 = len(envs)


def _make_env(ascension: int) -> tuple[slai.GameEnv, slai.GameState]:
    env = slai.GameEnv(ascension=ascension, fast_mode=FAST_MODE)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))
    return env, obs


def make_env_pool(
    num_envs: int,
    gamma: float,
    ascension: float = ASCENSION_LEVEL,
    ascension_max: int = ASCENSION_LEVEL,
    idx_offset: int = 0,
    total_envs: int = 0,
) -> EnvPool:
    # Episode stats live on the pool (not in _collect_rollout) so episodes spanning
    # rollout boundaries report true totals.
    total = total_envs or num_envs
    envs, obs = [], []
    for i in range(num_envs):
        env, ob = _make_env(_env_level(ascension, idx_offset + i, total, ascension_max))
        envs.append(env)
        obs.append(ob)
    return EnvPool(
        envs=envs,
        obs=obs,
        ep_rewards=[np.zeros(len(REWARD_STREAMS)) for _ in range(num_envs)],
        ep_lengths=[0] * num_envs,
        completed=[],
        gamma=gamma,
        ascension_c=ascension,
        ascension_max=ascension_max,
        idx_offset=idx_offset,
        total_envs=total,
    )


def legal_actions(pool: EnvPool) -> list[list[slai.Action]]:
    return [env.get_legal_actions() for env in pool.envs]


def step_env(pool: EnvPool, env_idx: int, action: slai.Action) -> tuple[np.ndarray, bool]:
    prev = pool.obs[env_idx]
    nxt, terminated = pool.envs[env_idx].step(action)
    reward = compute_reward(prev, nxt, terminated, pool.gamma)  # (K,)
    pool.ep_rewards[env_idx] += reward
    pool.ep_lengths[env_idx] += 1
    if terminated:
        # nxt is still the pre-reset terminal snapshot here
        pool.completed.append(
            EpisodeStats(
                pool.ep_rewards[env_idx],
                pool.ep_lengths[env_idx],
                won=nxt.character.health > 0,
                floor=cumulative_floor(nxt),
            )
        )
        pool.ep_rewards[env_idx] = np.zeros(len(REWARD_STREAMS))
        pool.ep_lengths[env_idx] = 0
        pool.envs[env_idx], nxt = _make_env(
            _env_level(
                pool.ascension_c,
                pool.idx_offset + env_idx,
                pool.total_envs or len(pool.envs),
                pool.ascension_max,
            )
        )
    pool.obs[env_idx] = nxt
    return reward, terminated


def drain_completed(pool: EnvPool) -> list[EpisodeStats]:
    completed, pool.completed = pool.completed, []
    return completed


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE over E parallel envs. Rewards/values are (T, E, K) with one GAE
    recursion per reward stream (GAE is linear in rewards, so the per-stream advantages
    sum to the single-critic advantage on the summed reward); `bootstrap` is (E, K) and
    `dones` is (T, E, 1), broadcast over streams. Each env's column is an independent
    trajectory and `dones` cut episodes within it — a terminal step zeroes the
    next-value term, so nothing leaks across an episode boundary (including when an env
    reset mid-rollout). Returns (returns, advantages), each (T, E, K)."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    for t in reversed(range(T)):
        next_value = bootstrap if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    return advantages + values, advantages


def _collect_rollout(
    model: ActorCritic,
    pool: EnvPool,
    rollout_length: int,
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[RolloutBuffer, list[EpisodeStats], dict[str, float]]:
    """Collect a fixed-length rollout into a columnar RolloutBuffer (rows = t*E + e),
    written in place into storage preallocated at (N, ...) — no per-step tree
    retention or torch.cat. GAE is computed per env (column) before flattening;
    advantages are normalized."""
    E = len(pool.envs)
    T = rollout_length
    N = T * E
    K = len(REWARD_STREAMS)
    buf_x = buf_mb = None  # allocated from the first step's shapes
    idx_at = torch.empty(N, dtype=torch.long, device=device)
    idx_l1 = torch.empty(N, dtype=torch.long, device=device)
    idx_l2 = torch.empty(N, dtype=torch.long, device=device)
    log_probs = torch.empty(N, device=device)
    values = torch.empty(T, E, K, device=device)
    rewards = torch.empty(T, E, K, device=device)
    dones = torch.empty(T, E, device=device)
    # Rest-site preference (XIV pathology: p(upgrade)=0 was invisible in scalars) —
    # device-side counters, read once after the loop (no per-step sync).
    rest_picks = torch.zeros((), device=device)
    upgrade_picks = torch.zeros((), device=device)

    # Set model to eval, turn off gradient computation
    model.eval()
    with torch.no_grad():
        for t in range(T):
            views = pool.obs
            legal = legal_actions(pool)
            x = encode_batch_game_state(views, device)
            mb = build_masks(views, legal, device)
            out, t_values = model(x, mb, greedy=False)

            if buf_x is None:
                buf_x = x.new_empty(N)
                buf_mb = mb.new_empty(N)
            rows = slice(t * E, (t + 1) * E)
            buf_x[rows] = x
            buf_mb[rows] = mb
            idx_at[rows] = out.idxs[:, Level.ACTION_TYPE]
            idx_l1[rows] = out.idxs[:, Level.L1]
            idx_l2[rows] = out.idxs[:, Level.L2]
            log_probs[rows] = out.log_prob.sum(-1)
            values[t] = t_values  # (E, K)

            # Campfire decision: both Rest and CardUpgrade legal — which was chosen?
            both = mb.mask_action_type[:, _AT_REST] & mb.mask_action_type[:, _AT_UPGRADE]
            chosen = out.idxs[:, Level.ACTION_TYPE]
            rest_picks += (both & (chosen == _AT_REST)).sum()
            upgrade_picks += (both & (chosen == _AT_UPGRADE)).sum()

            # Decode actions on CPU once (one device->host transfer, not per-env), then step.
            out_cpu = out.cpu()
            for i in range(E):
                reward, done = step_env(pool, i, action_from_actiontype(out_cpu, i, views[i]))
                rewards[t, i] = torch.from_numpy(reward)
                dones[t, i] = float(done)

        # Bootstrap value V(s_T) per env (a fresh env's value if it reset on the last step;
        # GAE's done-masking ensures that only contributes to non-terminal timesteps).
        views = pool.obs
        legal = legal_actions(pool)
        _, bootstrap = model(
            encode_batch_game_state(views, device), build_masks(views, legal, device), greedy=True
        )  # bootstrap (E, K)

    # Set model to training mode
    model.train()

    returns, advantages = _compute_gae(rewards, values, dones.unsqueeze(-1), bootstrap, gamma, lam)
    # Policy advantage: per-stream advantages summed (≡ single-critic GAE on the
    # summed reward, by linearity), then normalized as before.
    advantages = advantages.sum(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    n_campfire_picks = (rest_picks + upgrade_picks).item()
    rollout_info = {
        "rollout/terminations": dones.sum().item(),
        "rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
        # 0.0 also when no campfire was seen this rollout; XIV's frozen policy sat at 0 here.
        "rollout/rest_upgrade_rate": upgrade_picks.item() / n_campfire_picks
        if n_campfire_picks
        else 0.0,
    }

    return (
        RolloutBuffer(
            game_state=buf_x,
            mask_batch=buf_mb,
            idx_at=idx_at,
            idx_l1=idx_l1,
            idx_l2=idx_l2,
            log_probs_old=log_probs,
            values=values.reshape(-1, K),
            returns=returns.reshape(-1, K),
            advantages=advantages.reshape(-1, 1),
            batch_size=[N],
        ),
        drain_completed(pool),
        rollout_info,
    )


# ---- Sharded rollout (config rollout_workers > 0) ----
# The rollout's CPU work (engine step + reward + encode + masks) is sharded
# across worker processes; the model forward and sampling stay central. Workers
# write encoded observations/masks straight into one SharedMemory block (no
# pickling — the pipe transfer is what sank the 2026-07 overlap probe), and
# receive the sampled [ActionType, L1, L2] ints back through the same block
# (the engine-Action back-map needs the pre-step state, which the worker holds).
# With double_buffer, envs split into two gangs: the GPU forwards one gang
# while the other gang's workers step+encode (Sample Factory's double-buffered
# sampling; exactly on-policy, staleness 0).

_OP_STOP, _OP_ENCODE, _OP_STEP_ENCODE = 0, 1, 2


def _obs_tensor_items(x: TGameState) -> list[tuple[str, torch.Tensor]]:
    """Deterministic (name, tensor) walk of the observation — the shared-memory
    field layout. Padded pairs flatten to `.x`/`.mask`."""
    items: list[tuple[str, torch.Tensor]] = []
    for name in (
        "cards",
        "relics",
        "potions",
        "monsters",
        "event_options",
        "character",
    ):
        padded: TPadded = getattr(x, name)
        items.append((f"{name}.x", padded.x))
        items.append((f"{name}.mask", padded.mask))
    for name in (
        "energy",
        "map_grid",
        "room_node_idx",
        "room_mask",
        "map_meta",
        "screen",
        "ascension",
        "reward_meta",
        "shop_meta",
        "event_meta",
        "shop_card_prices",
        "shop_relic_prices",
        "shop_potion_prices",
    ):
        items.append((name, getattr(x, name)))
    return items


def _mask_tensor_items(mb: TMask) -> list[tuple[str, torch.Tensor]]:
    items = [("mask_action_type", mb.mask_action_type)]
    for key in sorted(mb.mask_action_idx.keys(), key=int):
        items.append((f"l1.{key}", mb.mask_action_idx[key]))
    items.append(("mask_target_card", mb.mask_target_card))
    items.append(("mask_target_potion", mb.mask_target_potion))
    return items


@dataclass
class ShardLayout:
    fields: dict[str, tuple[int, tuple[int, ...], np.dtype]]  # name -> (offset, shape, dtype)
    nbytes: int


def _build_shard_layout(num_envs: int, num_workers: int) -> ShardLayout:
    """Size every shared field from a 1-env probe encode (dims are config-derived;
    a fresh throwaway env keeps this layout-source-of-truth-free)."""
    env = slai.GameEnv(ascension=0, fast_mode=FAST_MODE)
    obs = env.reset(seed=0)
    cpu = torch.device("cpu")
    probe_x = encode_batch_game_state([obs], cpu)
    probe_mb = build_masks([obs], [env.get_legal_actions()], cpu)

    fields: dict[str, tuple[int, tuple[int, ...], np.dtype]] = {}
    offset = 0

    def add(name: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
        nonlocal offset
        offset = (offset + 63) // 64 * 64  # 64B-align every field
        fields[name] = (offset, shape, dtype)
        offset += int(np.prod(shape)) * dtype.itemsize

    for name, t in _obs_tensor_items(probe_x) + _mask_tensor_items(probe_mb):
        np_dtype = np.dtype(str(t.dtype).removeprefix("torch."))
        add(name, (num_envs, *t.shape[1:]), np_dtype)
    add("_actions", (num_envs, 3), np.dtype(np.int64))
    add("_rewards", (num_envs, len(REWARD_STREAMS)), np.dtype(np.float32))
    add("_dones", (num_envs,), np.dtype(np.float32))
    add("_ctrl_c", (1,), np.dtype(np.float64))  # live curriculum position
    add("_ops", (num_workers,), np.dtype(np.int64))
    return ShardLayout(fields=fields, nbytes=offset)


def _np_views(shm: SharedMemory, layout: ShardLayout) -> dict[str, np.ndarray]:
    return {
        name: np.frombuffer(shm.buf, dtype=dtype, count=int(np.prod(shape)), offset=off).reshape(
            shape
        )
        for name, (off, shape, dtype) in layout.fields.items()
    }


def _snapshot_to_device(
    views: dict[str, np.ndarray], lo: int, hi: int, device: torch.device
) -> tuple[TGameState, TMask]:
    """Copy the shard rows [lo, hi) out of shared memory onto the device as a
    TGameState/TMask pair. The .to(device) copy IS the snapshot boundary — the
    workers only overwrite these rows after the next signal."""
    B = [hi - lo]

    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(views[name][lo:hi]).to(device)

    def padded(name: str) -> TPadded:
        return TPadded(t(f"{name}.x"), t(f"{name}.mask"), batch_size=B)

    x = TGameState(
        cards=padded("cards"),
        relics=padded("relics"),
        potions=padded("potions"),
        monsters=padded("monsters"),
        event_options=padded("event_options"),
        character=padded("character"),
        energy=t("energy"),
        map_grid=t("map_grid"),
        room_node_idx=t("room_node_idx"),
        room_mask=t("room_mask"),
        map_meta=t("map_meta"),
        screen=t("screen"),
        ascension=t("ascension"),
        reward_meta=t("reward_meta"),
        shop_meta=t("shop_meta"),
        event_meta=t("event_meta"),
        shop_card_prices=t("shop_card_prices"),
        shop_relic_prices=t("shop_relic_prices"),
        shop_potion_prices=t("shop_potion_prices"),
        batch_size=B,
    )
    from tensordict import TensorDict

    mb = TMask(
        mask_action_type=t("mask_action_type"),
        mask_action_idx=TensorDict(
            {
                name.removeprefix("l1."): t(name)
                for name in views
                if name.startswith("l1.")
            },
            batch_size=B,
        ),
        mask_target_card=t("mask_target_card"),
        mask_target_potion=t("mask_target_potion"),
        batch_size=B,
    )
    return x, mb


def _shard_worker(
    worker_id: int,
    lo: int,
    hi: int,
    total_envs: int,
    seed: int,
    gamma: float,
    ascension_start: float,
    ascension_max: int,
    shm_name: str,
    layout: ShardLayout,
    ep_queue,
    req,
    done,
) -> None:
    """One rollout shard: owns envs [lo, hi), steps them on _OP_STEP_ENCODE using
    the action ints in shared memory, then encodes obs + masks for the shard into
    its rows. _OP_ENCODE encodes without stepping (rollout priming)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.set_num_threads(1)  # workers are the parallelism; don't oversubscribe
    shm = SharedMemory(name=shm_name)
    views = _np_views(shm, layout)
    cpu = torch.device("cpu")
    pool = make_env_pool(
        hi - lo,
        gamma,
        ascension=ascension_start,
        ascension_max=ascension_max,
        idx_offset=lo,
        total_envs=total_envs,
    )
    actions = views["_actions"]
    rewards = views["_rewards"]
    dones = views["_dones"]

    while True:
        req.wait()
        req.clear()
        op = int(views["_ops"][worker_id])
        if op == _OP_STOP:
            break
        pool.ascension_c = float(views["_ctrl_c"][0])  # promotions apply on reset
        if op == _OP_STEP_ENCODE:
            for j in range(hi - lo):
                at, l1, l2 = actions[lo + j]
                action = engine_action(int(at), int(l1), int(l2), pool.obs[j])
                reward, terminated = step_env(pool, j, action)
                rewards[lo + j] = reward
                dones[lo + j] = float(terminated)
        legal = legal_actions(pool)
        x = encode_batch_game_state(pool.obs, cpu)
        mb = build_masks(pool.obs, legal, cpu)
        for name, t in _obs_tensor_items(x) + _mask_tensor_items(mb):
            views[name][lo:hi] = t.numpy()
        for stats in drain_completed(pool):
            ep_queue.put(stats)
        done.set()

    # Release every numpy view before closing: frombuffer holds buffer exports,
    # and mmap refuses to close while any exist.
    del actions, rewards, dones, views
    shm.close()


@dataclass
class ShardedRollout:
    procs: list
    reqs: list
    dones: list
    ranges: list[tuple[int, int]]  # per-worker [lo, hi)
    gangs: list[list[int]]  # worker-index groups (1 = serial forward, 2 = double buffer)
    shm: SharedMemory
    views: dict[str, np.ndarray]
    ep_queue: object
    num_envs: int
    primed: bool = False


def make_sharded_rollout(
    num_envs: int,
    num_workers: int,
    gamma: float,
    ascension_start: float,
    ascension_max: int,
    double_buffer: bool,
    seed: int,
) -> ShardedRollout:
    gang_count = 2 if double_buffer else 1
    assert num_workers % gang_count == 0, "double_buffer needs an even worker count"
    assert num_envs % num_workers == 0, "num_envs must divide evenly across workers"
    layout = _build_shard_layout(num_envs, num_workers)
    shm = SharedMemory(create=True, size=layout.nbytes)
    views = _np_views(shm, layout)
    views["_ctrl_c"][0] = ascension_start

    ctx = mp.get_context("spawn")  # CUDA is live in this process; fork is unsafe
    ep_queue = ctx.Queue()
    per = num_envs // num_workers
    procs, reqs, dones, ranges = [], [], [], []
    for w in range(num_workers):
        lo, hi = w * per, (w + 1) * per
        req, done = ctx.Event(), ctx.Event()
        proc = ctx.Process(
            target=_shard_worker,
            args=(
                w,
                lo,
                hi,
                num_envs,
                seed * 7919 + w,
                gamma,
                ascension_start,
                ascension_max,
                shm.name,
                layout,
                ep_queue,
                req,
                done,
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)
        reqs.append(req)
        dones.append(done)
        ranges.append((lo, hi))
    per_gang = num_workers // gang_count
    gangs = [list(range(g * per_gang, (g + 1) * per_gang)) for g in range(gang_count)]
    return ShardedRollout(
        procs=procs,
        reqs=reqs,
        dones=dones,
        ranges=ranges,
        gangs=gangs,
        shm=shm,
        views=views,
        ep_queue=ep_queue,
        num_envs=num_envs,
    )


def _gang_range(sr: ShardedRollout, gang: list[int]) -> tuple[int, int]:
    return sr.ranges[gang[0]][0], sr.ranges[gang[-1]][1]


def _signal(sr: ShardedRollout, gang: list[int], op: int) -> None:
    for w in gang:
        sr.views["_ops"][w] = op
        sr.reqs[w].set()


def _wait(sr: ShardedRollout, gang: list[int]) -> None:
    for w in gang:
        # Bounded wait + liveness check: a crashed worker must surface as an
        # error here, never as a silent hang on a dead Event.
        while not sr.dones[w].wait(timeout=10.0):
            if not sr.procs[w].is_alive():
                raise RuntimeError(
                    f"rollout worker {w} died (exitcode {sr.procs[w].exitcode}); "
                    "see its traceback above"
                )
        sr.dones[w].clear()


def close_sharded_rollout(sr: ShardedRollout) -> None:
    for w in range(len(sr.procs)):
        sr.views["_ops"][w] = _OP_STOP
        sr.reqs[w].set()
    for proc in sr.procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    sr.views.clear()  # release buffer exports before mmap close (BufferError otherwise)
    sr.shm.close()
    sr.shm.unlink()


def _drain_shard_episodes(sr: ShardedRollout) -> list[EpisodeStats]:
    completed = []
    while not sr.ep_queue.empty():
        completed.append(sr.ep_queue.get())
    return completed


def _collect_rollout_sharded(
    model: ActorCritic,
    sr: ShardedRollout,
    rollout_length: int,
    gamma: float,
    lam: float,
    device: torch.device,
    ascension_c: float,
) -> tuple[RolloutBuffer, list[EpisodeStats], dict[str, float]]:
    """Sharded (and optionally double-buffered) counterpart of _collect_rollout.
    Buffer layout, GAE, and sampling are identical to the serial path — only
    where the CPU work runs differs."""
    E = sr.num_envs
    T = rollout_length
    N = T * E
    K = len(REWARD_STREAMS)
    sr.views["_ctrl_c"][0] = ascension_c
    buf_x = buf_mb = None
    idx_at = torch.empty(N, dtype=torch.long, device=device)
    idx_l1 = torch.empty(N, dtype=torch.long, device=device)
    idx_l2 = torch.empty(N, dtype=torch.long, device=device)
    log_probs = torch.empty(N, device=device)
    values = torch.empty(T, E, K, device=device)
    rewards = torch.empty(T, E, K, device=device)
    dones = torch.empty(T, E, device=device)
    rest_picks = torch.zeros((), device=device)
    upgrade_picks = torch.zeros((), device=device)

    model.eval()
    with torch.no_grad():
        if not sr.primed:
            for gang in sr.gangs:
                _signal(sr, gang, _OP_ENCODE)
            for gang in sr.gangs:
                _wait(sr, gang)
            sr.primed = True

        snaps = [
            _snapshot_to_device(sr.views, *_gang_range(sr, gang), device) for gang in sr.gangs
        ]

        def forward_and_dispatch(g: int, t: int) -> None:
            nonlocal buf_x, buf_mb
            lo, hi = _gang_range(sr, sr.gangs[g])
            x, mb = snaps[g]
            out, t_values = model(x, mb, greedy=False)
            if buf_x is None:
                buf_x = x.new_empty(N)
                buf_mb = mb.new_empty(N)
            rows = slice(t * E + lo, t * E + hi)
            buf_x[rows] = x
            buf_mb[rows] = mb
            idx_at[rows] = out.idxs[:, Level.ACTION_TYPE]
            idx_l1[rows] = out.idxs[:, Level.L1]
            idx_l2[rows] = out.idxs[:, Level.L2]
            log_probs[rows] = out.log_prob.sum(-1)
            values[t, lo:hi] = t_values

            both = mb.mask_action_type[:, _AT_REST] & mb.mask_action_type[:, _AT_UPGRADE]
            chosen = out.idxs[:, Level.ACTION_TYPE]
            rest_picks.add_((both & (chosen == _AT_REST)).sum())
            upgrade_picks.add_((both & (chosen == _AT_UPGRADE)).sum())

            sr.views["_actions"][lo:hi] = out.idxs.cpu().numpy()
            _signal(sr, sr.gangs[g], _OP_STEP_ENCODE)

        def collect_and_snapshot(g: int, t: int) -> None:
            lo, hi = _gang_range(sr, sr.gangs[g])
            _wait(sr, sr.gangs[g])
            rewards[t, lo:hi] = torch.from_numpy(sr.views["_rewards"][lo:hi].copy()).to(device)
            dones[t, lo:hi] = torch.from_numpy(sr.views["_dones"][lo:hi].copy()).to(device)
            snaps[g] = _snapshot_to_device(sr.views, lo, hi, device)

        # Ping-pong: gang g steps (in its workers) while the OTHER gang runs its
        # forward on the GPU. With one gang this degrades to strict alternation.
        for t in range(T):
            for g in range(len(sr.gangs)):
                forward_and_dispatch(g, t)
            for g in range(len(sr.gangs)):
                collect_and_snapshot(g, t)

        # Bootstrap V(s_T) per gang from the post-loop snapshots.
        bootstrap = torch.empty(E, K, device=device)
        for g, gang in enumerate(sr.gangs):
            lo, hi = _gang_range(sr, gang)
            _, boot = model(*snaps[g], greedy=True)
            bootstrap[lo:hi] = boot
    model.train()

    returns, advantages = _compute_gae(rewards, values, dones.unsqueeze(-1), bootstrap, gamma, lam)
    advantages = advantages.sum(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n_campfire_picks = (rest_picks + upgrade_picks).item()
    rollout_info = {
        "rollout/terminations": dones.sum().item(),
        "rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
        "rollout/rest_upgrade_rate": upgrade_picks.item() / n_campfire_picks
        if n_campfire_picks
        else 0.0,
    }

    return (
        RolloutBuffer(
            game_state=buf_x,
            mask_batch=buf_mb,
            idx_at=idx_at,
            idx_l1=idx_l1,
            idx_l2=idx_l2,
            log_probs_old=log_probs,
            values=values.reshape(-1, K),
            returns=returns.reshape(-1, K),
            advantages=advantages.reshape(-1, 1),
            batch_size=[N],
        ),
        _drain_shard_episodes(sr),
        rollout_info,
    )


def _cpu_state_dict(model: ActorCritic) -> dict[str, torch.Tensor]:
    """Model weights as CPU tensors — picklable across processes on any backend (CUDA
    IPC works, but MPS has no portable cross-process sharing; CPU is safe for both and
    the GPU->CPU copy of ~9M params is negligible)."""
    return {k: v.cpu() for k, v in model.state_dict().items()}


def _rollout_worker(
    conn: Connection,
    model_config: dict,
    num_envs: int,
    rollout_length: int,
    gamma: float,
    lam: float,
    seed: int,
) -> None:
    """Side process: receive weights, collect one rollout, reply with the rollout (as
    CPU tensors) + episode stats. Envs are built here — engine objects aren't picklable.
    Data is collected with the weights of the PREVIOUS update (staleness 1); PPO's ratio
    clipping absorbs it, guarded by the approx_kl/clip_fraction logs."""
    # NOTE: the overlap worker's pool runs at the fixed ASCENSION_LEVEL default — the
    # soft curriculum does not propagate here (overlap_rollout is off; measured loss).
    torch.set_num_threads(1)  # master keeps the P-cores for the update
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = ActorCritic(**model_config)
    model.to(device)  # moves params AND non-persistent buffers (e.g. map_encoder._row_of_node)
    pool = make_env_pool(num_envs, gamma)

    while True:
        # Receive current weights (CPU tensors), load them, collect one rollout.
        state_dict = conn.recv()
        if state_dict is None:
            return
        model.load_state_dict(state_dict)
        buf, completed, rollout_info = _collect_rollout(
            model, pool, rollout_length, gamma, lam, device
        )
        # Hand the rollout back as CPU tensors — picklable across processes on any
        # backend (no portable CUDA-IPC / MPS cross-process sharing); the master moves
        # it onto its device. Simpler and correct vs the old shared-memory handoff.
        conn.send((buf.cpu(), completed, rollout_info))


def _minibatch_metrics(
    loss_policy: torch.Tensor,
    loss_value: torch.Tensor,
    loss_entropy: torch.Tensor,
    entropies: torch.Tensor,
    loss_kl_uniform: torch.Tensor,
    kl_uniform: torch.Tensor,
    ratio: torch.Tensor,
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    grad_norm: torch.Tensor,
    clip_eps: float,
) -> dict[str, float]:
    """Per-minibatch TensorBoard scalars (minibatch means). Schulman's approx-KL
    estimator; clip fraction = share of moved-off ratios; grad norm is the pre-clip
    total (otherwise discarded by clip_grad_norm_)."""
    ent_mean = entropies.mean(dim=0)
    klu_mean = kl_uniform.mean(dim=0)
    return {
        "loss/policy": loss_policy.item(),
        "loss/value": loss_value.item(),
        "loss/entropy": loss_entropy.item(),
        "loss/kl_uniform": loss_kl_uniform.item(),
        "entropy/L1": ent_mean[0].item(),
        "entropy/L2": ent_mean[1].item(),
        "entropy/L3": ent_mean[2].item(),
        "kl_uniform/L1": klu_mean[0].item(),
        "kl_uniform/L2": klu_mean[1].item(),
        "kl_uniform/L3": klu_mean[2].item(),
        "update/approx_kl": ((ratio - 1) - (log_probs_new - log_probs_old)).mean().item(),
        "update/clip_fraction": ((ratio - 1).abs() > clip_eps).float().mean().item(),
        "grad/norm_preclip": grad_norm.item(),
    }


def _update_ppo(
    model: ActorCritic,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    coef_value: float,
    coef_entropy: float,
    coef_kl_uniform: float,
    max_grad_norm: float,
    device: torch.device,
    target_kl: float | None = None,
    amp: bool = False,
) -> dict[str, float]:
    """One PPO update over the buffer. Returns iteration-mean metrics keyed by their
    TensorBoard scalar names."""
    records: list[dict[str, float]] = []

    # The decomposition's instrumentation: per-stream explained variance from
    # rollout-time values — which return stream the critic can predict (EV → 1)
    # and which carries the residual noise (typically the outcome stream).
    ev = 1.0 - (buffer.returns - buffer.values).var(dim=0) / (buffer.returns.var(dim=0) + 1e-8)
    epochs_run = 0
    for _ in range(num_epochs):
        epoch_start = len(records)
        ep = shuffle_rollout_buffer(buffer, device)
        for start in range(0, len(buffer), minibatch_size):
            rows = slice(start, start + minibatch_size)
            # bf16 autocast covers the model forward only (Ada ~2x matmul
            # throughput; no GradScaler needed for bf16). Loss math stays fp32 —
            # ratio = exp(Δlogprob) is precision-sensitive — so outputs are
            # upcast at the boundary. Softmax/log_softmax run fp32 under
            # autocast policy regardless.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                log_probs_new, entropies, kl_uniform, values_new = model.evaluate_actions(
                    ep.game_state[rows],
                    ep.mask_batch[rows],
                    ep.idx_at[rows],
                    ep.idx_l1[rows],
                    ep.idx_l2[rows],
                )
            if amp:
                log_probs_new = log_probs_new.float()
                entropies = entropies.float()
                kl_uniform = kl_uniform.float()
                values_new = values_new.float()
            log_probs_old = ep.log_probs_old[rows]
            adv = ep.advantages[rows].squeeze(-1)
            returns = ep.returns[rows]
            values_old = ep.values[rows]

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            loss_policy = -torch.mean(torch.min(surr1, surr2))

            # Value loss: summed over streams, mean over the batch — keeps each
            # stream's gradient at the single-critic scale (a mean over K would
            # implicitly divide coef_value by K).
            if clip_value_loss:
                values_clipped = values_old + torch.clamp(
                    values_new - values_old, -clip_eps, clip_eps
                )
                lv_unclipped = torch.pow(values_new - returns, 2)
                lv_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.max(lv_unclipped, lv_clipped).sum(dim=-1).mean()
            else:
                # 0.5·MSE to match the clipped branch's canonical PPO scale; without it,
                # toggling clip_value_loss silently 2x'd the effective critic weight.
                loss_value = 0.5 * F.mse_loss(values_new, returns, reduction="none").sum(-1).mean()

            loss_entropy = -torch.mean(entropies.sum(dim=-1))
            # Uniform-KL barrier, summed per head like the entropy term — but unlike
            # entropy it can't be satisfied by the cheapest head (each head's KL only
            # falls by lifting its own tail probs) and must never decay: it is what
            # keeps rare actions reachable once the policy peaks (XIV's L1 collapse).
            loss_kl_uniform = torch.mean(kl_uniform.sum(dim=-1))
            loss = (
                loss_policy
                + coef_value * loss_value
                + coef_entropy * loss_entropy
                + coef_kl_uniform * loss_kl_uniform
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            records.append(
                _minibatch_metrics(
                    loss_policy,
                    loss_value,
                    loss_entropy,
                    entropies,
                    loss_kl_uniform,
                    kl_uniform,
                    ratio,
                    log_probs_new,
                    log_probs_old,
                    grad_norm,
                    clip_eps,
                )
            )
        epochs_run += 1
        # Target-KL early-stop (XIV fix #3): the clip bounds each sample's incentive,
        # not the cumulative drift across epochs on frozen log_probs_old — stop
        # squeezing this batch once realized movement exceeds the trust region.
        if target_kl is not None:
            epoch_kl = sum(r["update/approx_kl"] for r in records[epoch_start:]) / max(
                1, len(records) - epoch_start
            )
            if epoch_kl > 1.5 * target_kl:
                break
    metrics = {key: sum(r[key] for r in records) / len(records) for key in records[0]}
    metrics["update/epochs_run"] = float(epochs_run)
    for k, name in enumerate(REWARD_STREAMS):
        metrics[f"Value/ev_{name}"] = ev[k].item()
    return metrics


def _save_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    total_steps: int,
    curriculum_c: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "total_steps": total_steps,
            "curriculum_c": curriculum_c,
        },
        path,
    )


def train(
    exp_name: str,
    num_iterations: int,
    log_every: int,
    save_every: int,
    model: ActorCritic,
    model_config: dict,
    optimizer: torch.optim.Optimizer,
    rollout_length: int,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    gamma: float,
    lam: float,
    coef_value: float,
    coef_entropy_max: float,
    coef_entropy_min: float,
    entropy_decay_steps: float,
    coef_kl_uniform: float,
    max_grad_norm: float,
    num_envs: int,
    ascension_start: float,
    ascension_max: int,
    promote_win_rate: float,
    promote_streak: int,
    curriculum_step: float,
    target_kl: float,
    device: torch.device,
    start_iteration: int = 0,
    total_steps: int = 0,
    overlap_rollout: bool = False,
    amp: bool = False,
    rollout_workers: int = 0,
    double_buffer: bool = False,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)
    # A resumed optimizer's state loads on CPU (_load_checkpoint runs before this
    # model.to), so realign it to the model's device or Adam's fused step mismatches
    # params (cuda) against exp_avg (cpu). No-op on a fresh run (empty state).
    for opt_state in optimizer.state.values():
        for key, val in opt_state.items():
            if isinstance(val, torch.Tensor):
                opt_state[key] = val.to(device)
    ckpt_path = f"experiments/{exp_name}/checkpoint.pth"
    iteration = start_iteration  # for the interrupt save, if it fires pre-loop
    c = ascension_start  # soft-curriculum position (see _curriculum_step)
    streak = 0

    # Eval runs a fixed-seed greedy battery in a persistent side process (mirrors the
    # rollout worker): weights are sent on the save cadence and the aggregate metrics
    # drained when ready, never blocking the training loop.
    eval_ctx = mp.get_context("spawn")
    eval_conn, eval_child = eval_ctx.Pipe()
    eval_proc = eval_ctx.Process(
        target=eval_battery_worker, args=(eval_child, model_config, gamma), daemon=True
    )
    eval_proc.start()
    eval_busy = False

    # Overlapped mode: a worker collects rollout(t+1) while we update on rollout(t);
    # serial mode keeps the envs in-process (the A/B reference for the overlap flag).
    if overlap_rollout:
        worker_conn, child_conn = eval_ctx.Pipe()
        rollout_worker = eval_ctx.Process(
            target=_rollout_worker,
            args=(
                child_conn,
                model_config,
                num_envs,
                rollout_length,
                gamma,
                lam,
                random.randint(0, 2**31 - 1),
            ),
            daemon=True,
        )
        rollout_worker.start()
        worker_conn.send(_cpu_state_dict(model))  # kick off the first rollout
    elif rollout_workers > 0:
        sharded = make_sharded_rollout(
            num_envs,
            rollout_workers,
            gamma,
            ascension_start=c,
            ascension_max=ascension_max,
            double_buffer=double_buffer,
            seed=random.randint(0, 2**31 - 1),
        )
    else:
        pool = make_env_pool(num_envs, gamma, ascension=c, ascension_max=ascension_max)

    try:
        for iteration in range(start_iteration, num_iterations):
            # Entropy coef keyed to env steps (not iterations), linear decay to the
            # floor — consistent across resumes since it derives from total_steps.
            frac = min(1.0, total_steps / entropy_decay_steps)
            coef_entropy = coef_entropy_max + frac * (coef_entropy_min - coef_entropy_max)
            t_start = time.perf_counter()
            if overlap_rollout:
                # Receive the rollout the worker collected during the last update (CPU
                # tensors), move it to our device, restart the worker on fresh weights.
                buf_cpu, completed, rollout_info = worker_conn.recv()
                buffer = buf_cpu.to(device)
                worker_conn.send(_cpu_state_dict(model))
            elif rollout_workers > 0:
                buffer, completed, rollout_info = _collect_rollout_sharded(
                    model, sharded, rollout_length, gamma, lam, device, c
                )
            else:
                buffer, completed, rollout_info = _collect_rollout(
                    model, pool, rollout_length, gamma, lam, device
                )
            t_rollout = time.perf_counter()
            total_steps += rollout_length * num_envs
            metrics = _update_ppo(
                model,
                buffer,
                optimizer,
                num_epochs,
                minibatch_size,
                clip_eps,
                clip_value_loss,
                coef_value,
                coef_entropy,
                coef_kl_uniform,
                max_grad_norm,
                device,
                target_kl,
                amp,
            )
            t_update = time.perf_counter()
            # Eval battery (persistent worker): drain a finished one and, on the save
            # cadence when idle, launch a new one - both non-blocking. Logged at eval_iter.
            if eval_busy and eval_conn.poll():
                eval_iter, eval_lvl, eval_metrics = eval_conn.recv()
                eval_busy = False
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, eval_iter)
                writer.add_scalar("eval/ascension", eval_lvl, eval_iter)
                print(
                    f"  eval@{eval_iter} A{eval_lvl}: win_rate={eval_metrics[EVAL_WIN_RATE]:.3f} "
                    f"floor={eval_metrics[EVAL_FLOOR_AVG]:.2f} "
                    f"act2={eval_metrics[EVAL_ACT2_RATE]:.2f} "
                    f"reward={eval_metrics[EVAL_REWARD_AVG]:.4f}"
                )
                c_prev = c
                c, streak = _curriculum_step(
                    c,
                    streak,
                    eval_metrics[EVAL_WIN_RATE],
                    eval_lvl,
                    promote_win_rate,
                    promote_streak,
                    curriculum_step,
                    ascension_max,
                )
                if c != c_prev:
                    if not overlap_rollout:
                        pool.ascension_c = c  # envs graduate on their next reset
                    print(f"  curriculum: c={c:.2f}, frontier A{_eval_level(c, ascension_max)}")
            if iteration % save_every == 0 and not eval_busy:
                eval_conn.send((_cpu_state_dict(model), iteration, _eval_level(c, ascension_max)))
                eval_busy = True
            if iteration % log_every == 0:
                print(
                    f"Iter {iteration} | steps={total_steps} | "
                    f"policy={metrics['loss/policy']:.4f} value={metrics['loss/value']:.4f} | "
                    f"episodes={len(completed)}"
                )
                for key, value in metrics.items():
                    writer.add_scalar(key, value, iteration)
                writer.add_scalar("entropy/coef", coef_entropy, iteration)
                writer.add_scalar("train/ascension", c, iteration)
                writer.add_scalar("Steps/total", total_steps, iteration)
                writer.add_scalar("Time/rollout", t_rollout - t_start, iteration)
                writer.add_scalar("Time/update", t_update - t_rollout, iteration)
                for key, value in rollout_info.items():
                    writer.add_scalar(key, value, iteration)
                if completed:
                    for key, value in aggregate_episodes(completed).items():
                        writer.add_scalar(f"{PREFIX_TRAIN}{key}", value, iteration)
                    writer.add_scalar(f"{PREFIX_TRAIN}completed_count", len(completed), iteration)
                    for k, name in enumerate(REWARD_STREAMS):
                        writer.add_scalar(
                            f"{PREFIX_TRAIN}reward_avg_{name}",
                            sum(e.stream_rewards[k] for e in completed) / len(completed),
                            iteration,
                        )
                    # Floor-stream acceptance diagnostic: returns should be ordered by
                    # death depth (the analog of the XIV corr(finalHP, return) check).
                    floors = [e.floor for e in completed]
                    rets = [e.total_reward for e in completed]
                    if len(completed) >= 8 and np.std(floors) > 0 and np.std(rets) > 0:
                        writer.add_scalar(
                            f"{PREFIX_TRAIN}corr_floor_return",
                            float(np.corrcoef(floors, rets)[0, 1]),
                            iteration,
                        )
            if iteration % save_every == 0:
                _save_checkpoint(ckpt_path, model, optimizer, iteration, total_steps, c)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        _save_checkpoint(ckpt_path, model, optimizer, iteration, total_steps, c)
    finally:
        if overlap_rollout and rollout_worker.is_alive():
            try:
                worker_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        if rollout_workers > 0:
            close_sharded_rollout(sharded)
        if eval_proc.is_alive():
            try:
                eval_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
    writer.close()


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def _load_checkpoint(
    ckpt_path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> tuple[int, int, float | None]:
    """Resume model/optimizer/RNG from `ckpt_path` if it exists; returns
    (start_iteration, total_steps, curriculum_c), or (0, 0, None) for a fresh run."""
    if not os.path.exists(ckpt_path):
        return 0, 0, None

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iteration = ckpt["iteration"] + 1
    total_steps = ckpt["total_steps"]
    # Offset the stream seeds so a resume doesn't replay the run's env-seed sequence
    random.seed(seed + start_iteration)
    torch.manual_seed(seed + start_iteration)
    print(f"Resuming from {ckpt_path}: iteration {start_iteration}, {total_steps} steps")
    return start_iteration, total_steps, ckpt.get("curriculum_c")


if __name__ == "__main__":
    # nohup-backgrounded processes ignore SIGINT; route SIGTERM into the same
    # KeyboardInterrupt path so `kill <pid>` checkpoints the current iteration.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    # Optional config override (e.g. config.smoke.yml for the R6 smoke run)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)

    # TF32 on Ampere+ GPUs: near-free matmul throughput (no-op on CPU/MPS)
    torch.set_float32_matmul_precision("high")

    # Set seeds
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Instance `ActorCritic` model and optimizer
    model = ActorCritic(**config["model"])
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])
    os.makedirs(f"experiments/{config['exp_name']}", exist_ok=True)
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    # Load checkpoint
    ckpt_path = f"experiments/{config['exp_name']}/checkpoint.pth"
    start_iteration, total_steps, curriculum_c = _load_checkpoint(
        ckpt_path, model, optimizer, seed
    )

    print(f"Starting training: {config['exp_name']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  num_envs={config['num_envs']}, rollout_length={config['rollout_length']}")
    train(
        exp_name=config["exp_name"],
        num_iterations=int(config["num_iterations"]),
        log_every=config["log_every"],
        save_every=config["save_every"],
        model=model,
        model_config=config["model"],
        optimizer=optimizer,
        rollout_length=config["rollout_length"],
        num_epochs=config["num_epochs"],
        minibatch_size=config["minibatch_size"],
        clip_eps=config["clip_eps"],
        clip_value_loss=config["clip_value_loss"],
        gamma=config["gamma"],
        lam=config["lam"],
        coef_value=config["coef_value"],
        coef_entropy_max=config["coef_entropy_max"],
        coef_entropy_min=config["coef_entropy_min"],
        entropy_decay_steps=float(config["entropy_decay_steps"]),
        coef_kl_uniform=config["coef_kl_uniform"],
        max_grad_norm=config["max_grad_norm"],
        num_envs=config["num_envs"],
        ascension_start=(
            float(config["ascension_start"]) if curriculum_c is None else curriculum_c
        ),
        ascension_max=int(config["ascension_max"]),
        promote_win_rate=config["promote_win_rate"],
        promote_streak=int(config["promote_streak"]),
        curriculum_step=config["curriculum_step"],
        target_kl=config["target_kl"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        start_iteration=start_iteration,
        total_steps=total_steps,
        overlap_rollout=bool(config.get("overlap_rollout", False)),
        amp=bool(config.get("amp", False)),
        rollout_workers=int(config.get("rollout_workers", 0)),
        double_buffer=bool(config.get("double_buffer", False)),
    )
