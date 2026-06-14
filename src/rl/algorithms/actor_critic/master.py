"""PPO training with fixed-length rollouts and in-process environments.

Envs run in-process with `fast_mode=True` (the engine auto-advances trivial
single-legal-action states, so the trainer only sees real choice points).
`env.step` returns a 2-tuple `(obs, terminated)`; reward is computed trainer-side.

A rollout is stored columnar in a `RolloutBuffer`: the batched per-step tensors
(encoded state, masks, recorded action indices, log-probs, values) are written in
place into preallocated (N, ...) storage, and PPO minibatches index them by row.
Masks come from each env's `get_legal_actions()` (the authoritative legal set), so
recompute needs no live envs.
"""

import multiprocessing as mp
import os
import random
import shutil
import signal
import time
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np
import slai
import torch
import torch.multiprocessing  # noqa: F401 — registers tensor reductions (shm handles over pipes)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.types import TMask
from src.rl.action_space.masks import build_masks
from src.rl.types import action_from_actiontype
from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.types import TGameState
from src.rl.types import NUM_ACTION_TYPES
from src.rl.encoding.state import encode_batch_game_state
from src.rl.models import ActorCritic
from src.rl.reward import REWARD_STREAMS
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class RolloutBuffer:
    """Columnar rollout (N = rollout_length * num_envs rows). The full-batch masks are
    stored once and row-sliced per minibatch; GAE returns/advantages are precomputed.
    Values/returns are per reward stream (K = len(REWARD_STREAMS)); the policy
    advantage is the per-stream advantages summed, then normalized."""

    x_game_state: TGameState  # (N, ...)
    mask_batch: TMask  # (N, ...) — row-sliced per minibatch (mask_batch[rows])
    option_idx: torch.Tensor  # (N,) recorded L1 ActionType pick
    selection_idx: torch.Tensor  # (N,) recorded L2 entity pick
    target_idx: torch.Tensor  # (N,) recorded L3 monster pick (-1 if none)
    log_probs_old: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, K)
    returns: torch.Tensor  # (N, K)
    advantages: torch.Tensor  # (N, 1), summed over streams, normalized

    def __len__(self) -> int:
        return self.log_probs_old.shape[0]


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals
    length: int
    won: bool
    floor: int

    @property
    def total_reward(self) -> float:
        return float(self.stream_rewards.sum())


# =============================================================================
# Behavioral telemetry
# =============================================================================

# Behaviorally-loaded ActionTypes surfaced to TensorBoard (option_idx value ==
# int(ActionType)). Names are resolved against the engine enum at import, so a
# rename/removal in slai fails loudly here instead of silently mislabeling a tag.
_TELEMETRY_ACTION_NAMES = (
    "CardPurge",
    "ShopPurge",  # P4 deck thinning
    "ShopBuyCard",
    "ShopBuyRelic",
    "ShopBuyPotion",  # P3 shop spend
    "RewardTakeGold",
    "RewardTakePotion",
    "RewardTakeCard",
    "RewardTakeRelic",  # P2 loot
    "PotionUse",
    "PotionDiscard",  # P2 potions
    "Rest",
    "CardUpgrade",  # P1/P3 rest vs upgrade
    "RoomSelect",
    "TurnEnd",  # P3 routing / P5 turn end
)
_TELEMETRY_AT_IDX = {n: int(getattr(slai.ActionType, n)) for n in _TELEMETRY_ACTION_NAMES}
_REWARD_TAKE_IDX = [
    int(getattr(slai.ActionType, n))
    for n in ("RewardTakeGold", "RewardTakePotion", "RewardTakeCard", "RewardTakeRelic")
]
_ROOM_EXIT_IDX = int(slai.ActionType.RoomExit)
_POTION_AT_IDX = [int(slai.ActionType.PotionUse), int(slai.ActionType.PotionDiscard)]


def _behavioral_metrics(buffer: "RolloutBuffer") -> dict[str, float]:
    """Per-ActionType behavioral rates over a rollout (P1-P5 observability), all
    derived from already-stored buffer fields. These are SAMPLED actions, so noisy;
    the low-variance signal is the greedy eval battery. `cond_rate` = when the kind
    was legal, how often it was chosen; `avail_rate` = how often the choice arose."""
    opt = buffer.option_idx  # (N,) chosen L1 ActionType
    avail = buffer.mask_batch.mask_action_type  # (N, NUM_ACTION_TYPES) bool legality
    n = float(opt.shape[0])
    chosen = torch.bincount(opt, minlength=NUM_ACTION_TYPES).float()
    available = avail.sum(0).float()

    metrics: dict[str, float] = {}
    for name, idx in _TELEMETRY_AT_IDX.items():
        a = available[idx].item()
        metrics[f"Actions/avail_rate/{name}"] = a / n
        metrics[f"Actions/cond_rate/{name}"] = chosen[idx].item() / a if a > 0 else 0.0

    # Reward-skip: among steps where a RewardTake* was legal, fraction choosing RoomExit.
    reward_avail = avail[:, _REWARD_TAKE_IDX].any(dim=1)
    n_reward = reward_avail.sum().item()
    if n_reward > 0:
        skipped = ((opt == _ROOM_EXIT_IDX) & reward_avail).sum().item()
        metrics["Actions/reward_skip_rate"] = skipped / n_reward
    # Potion-held fraction: PotionUse/PotionDiscard are legal iff a potion is held.
    metrics["Actions/potion_held_frac"] = avail[:, _POTION_AT_IDX].any(dim=1).float().mean().item()
    return metrics


# =============================================================================
# Environment manager
# =============================================================================


class EnvironmentManager:
    def __init__(self, num_envs: int, gamma: float):
        self.num_envs = num_envs
        self._gamma = gamma
        self._envs: list[slai.GameEnv] = []
        self._obs: list[slai.GameState] = []
        # Episode stats live here (not in _collect_rollout) so episodes spanning
        # rollout boundaries report true totals.
        self._ep_rewards = [np.zeros(len(REWARD_STREAMS)) for _ in range(num_envs)]
        self._ep_lengths = [0] * num_envs
        self._completed: list[EpisodeStats] = []
        for _ in range(num_envs):
            env, obs = self._make_env()
            self._envs.append(env)
            self._obs.append(obs)

    @staticmethod
    def _make_env() -> tuple[slai.GameEnv, slai.GameState]:
        env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
        obs = env.reset(seed=random.randint(0, 2**31 - 1))
        return env, obs

    def get_view_states(self) -> list[slai.GameState]:
        return self._obs

    def get_legal_actions(self) -> list[list]:
        return [env.get_legal_actions() for env in self._envs]

    def step(self, env_idx: int, action) -> tuple[np.ndarray, bool]:
        prev = self._obs[env_idx]
        nxt, terminated = self._envs[env_idx].step(action)
        reward = compute_reward(prev, nxt, terminated, self._gamma)  # (K,)
        self._ep_rewards[env_idx] += reward
        self._ep_lengths[env_idx] += 1
        if terminated:
            # nxt is still the pre-reset terminal snapshot here
            self._completed.append(
                EpisodeStats(
                    self._ep_rewards[env_idx],
                    self._ep_lengths[env_idx],
                    won=nxt.character.health > 0,
                    floor=nxt.map.y_current or 0,
                )
            )
            self._ep_rewards[env_idx] = np.zeros(len(REWARD_STREAMS))
            self._ep_lengths[env_idx] = 0
            env, nxt = self._make_env()
            self._envs[env_idx] = env
        self._obs[env_idx] = nxt
        return reward, terminated

    def drain_completed(self) -> list[EpisodeStats]:
        completed, self._completed = self._completed, []
        return completed


# =============================================================================
# GAE
# =============================================================================


def _compute_gae(rewards, values, dones, bootstrap, gamma, lam):
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


# =============================================================================
# Rollout collection
# =============================================================================


def _collect_rollout(
    model: ActorCritic,
    env_mgr: EnvironmentManager,
    rollout_length: int,
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[RolloutBuffer, list[EpisodeStats]]:
    """Collect a fixed-length rollout into a columnar RolloutBuffer (rows = t*E + e),
    written in place into storage preallocated at (N, ...) — no per-step tree
    retention or torch.cat. GAE is computed per env (column) before flattening;
    advantages are normalized."""
    E = env_mgr.num_envs
    T = rollout_length
    N = T * E
    K = len(REWARD_STREAMS)
    buf_x = buf_mb = None  # allocated from the first step's shapes
    opt_idx = torch.empty(N, dtype=torch.long, device=device)
    sel_idx = torch.empty(N, dtype=torch.long, device=device)
    tgt_idx = torch.empty(N, dtype=torch.long, device=device)
    log_probs = torch.empty(N, device=device)
    values = torch.empty(T, E, K, device=device)
    rewards = torch.empty(T, E, K, device=device)
    dones = torch.empty(T, E, device=device)

    model.eval()
    with torch.no_grad():
        for t in range(T):
            views = env_mgr.get_view_states()
            legal = env_mgr.get_legal_actions()
            x = encode_batch_game_state(views, device)
            mb = build_masks(views, legal, device)
            out = model(x, mb, sample=True)

            if buf_x is None:
                buf_x = x.new_empty(N)
                buf_mb = mb.new_empty(N)
            rows = slice(t * E, (t + 1) * E)
            buf_x[rows] = x
            buf_mb[rows] = mb
            opt_idx[rows] = out.option.idx
            sel_idx[rows] = out.selection.idx
            tgt_idx[rows] = out.target.idx
            log_probs[rows] = out.total_log_prob()
            values[t] = out.values  # (E, K)

            # Extract the action ints once (not per-env .item()), then step each env.
            ops, sels, tgts = torch.stack(
                [out.option.idx, out.selection.idx, out.target.idx]
            ).tolist()
            for i in range(E):
                reward, done = env_mgr.step(i, action_from_actiontype(ops[i], sels[i], tgts[i]))
                rewards[t, i] = torch.from_numpy(reward)
                dones[t, i] = float(done)

        # Bootstrap value V(s_T) per env (a fresh env's value if it reset on the last step;
        # GAE's done-masking ensures that only contributes to non-terminal timesteps).
        views = env_mgr.get_view_states()
        legal = env_mgr.get_legal_actions()
        boot_out = model(
            encode_batch_game_state(views, device), build_masks(views, legal, device), sample=False
        )
        bootstrap = boot_out.values  # (E, K)
    model.train()

    returns, advantages = _compute_gae(rewards, values, dones.unsqueeze(-1), bootstrap, gamma, lam)
    # Policy advantage: per-stream advantages summed (≡ single-critic GAE on the
    # summed reward, by linearity), then normalized as before.
    advantages = advantages.sum(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    rollout_info = {
        "Rollout/terminations": dones.sum().item(),
        "Rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
    }

    return (
        RolloutBuffer(
            x_game_state=buf_x,
            mask_batch=buf_mb,
            option_idx=opt_idx,
            selection_idx=sel_idx,
            target_idx=tgt_idx,
            log_probs_old=log_probs,
            values=values.reshape(-1, K),
            returns=returns.reshape(-1, K),
            advantages=advantages.reshape(-1, 1),
        ),
        env_mgr.drain_completed(),
        rollout_info,
    )


# Greedy deterministic play can loop; cap so a hung eval can't wedge its worker
_EVAL_MAX_STEPS = 1000
_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # fixed test set => low-variance trend


def _run_eval_battery(model: ActorCritic, device: torch.device, gamma: float) -> dict[str, float]:
    """Greedy battery over a fixed seed set; returns outcome + behavioral metrics. Same
    encode->mask->forward path as training/watch. Fixed seeds make it a consistent test
    set across checkpoints (paired comparison; std/sqrt(N) variance, not the old 1/sqrt(1))."""
    rewards: list[float] = []
    lengths: list[int] = []
    wins: list[bool] = []
    floors: list[int] = []
    chosen = torch.zeros(NUM_ACTION_TYPES)
    available = torch.zeros(NUM_ACTION_TYPES)
    n_reward = 0
    n_reward_skip = 0
    with torch.no_grad():
        for seed in _EVAL_SEEDS:
            env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
            obs = env.reset(seed=seed)
            total_reward = 0.0
            length = 0
            terminated = False
            while not terminated and length < _EVAL_MAX_STEPS:
                legal = env.get_legal_actions()
                if not legal:
                    break
                x = encode_batch_game_state([obs], device)
                mb = build_masks([obs], [legal], device)
                out = model(x, mb, sample=False)
                m = mb.mask_action_type[0]
                at = int(out.option.idx[0].item())
                chosen[at] += 1.0
                available += m.float()
                if m[_REWARD_TAKE_IDX].any():
                    n_reward += 1
                    if at == _ROOM_EXIT_IDX:
                        n_reward_skip += 1
                action = out.get_action(0)
                prev = obs
                obs, terminated = env.step(action)
                total_reward += float(compute_reward(prev, obs, terminated, gamma).sum())
                length += 1
            rewards.append(total_reward)
            lengths.append(length)
            wins.append(bool(terminated and obs.character.health > 0))
            floors.append(obs.map.y_current or 0)

    n = len(rewards)
    metrics: dict[str, float] = {
        "Eval/win_rate": sum(wins) / n,
        "Eval/avg_floor": sum(floors) / n,
        "Eval/avg_length": sum(lengths) / n,
        "Eval/reward_mean": float(np.mean(rewards)),
        "Eval/reward_std": float(np.std(rewards)),
    }
    for name, idx in _TELEMETRY_AT_IDX.items():
        a = available[idx].item()
        metrics[f"Eval/cond_rate/{name}"] = chosen[idx].item() / a if a > 0 else 0.0
    if n_reward > 0:
        metrics["Eval/reward_skip_rate"] = n_reward_skip / n_reward
    return metrics


def _eval_battery_worker(conn, model_config, gamma) -> None:
    """Persistent eval worker (mirrors _rollout_worker's lifecycle): receive (weights,
    iteration), run the fixed-seed greedy battery, reply with metrics. `None` stops it.
    One thread, so it rides the spare core off the training critical path."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = ActorCritic(**model_config)
    model.eval()
    while True:
        msg = conn.recv()
        if msg is None:
            return
        state_dict, iteration = msg
        model.load_state_dict(state_dict)
        conn.send((iteration, _run_eval_battery(model, device, gamma)))


# =============================================================================
# Overlapped rollout collection (rollout t+1 runs while the master updates on t)
# =============================================================================


def _buffer_clone(buf: RolloutBuffer) -> RolloutBuffer:
    return RolloutBuffer(**{f.name: getattr(buf, f.name).clone() for f in fields(RolloutBuffer)})


def _rollout_worker(conn, model_config, num_envs, rollout_length, gamma, lam, seed) -> None:
    """Side process: receive weights, collect one rollout, expose it via a stable
    shared-memory buffer (sent as handles once), reply with episode stats. Envs are
    built here — engine objects aren't picklable. Data is collected with the weights
    of the PREVIOUS update (staleness 1); PPO's ratio clipping absorbs it, guarded by
    the approx_kl/clip_fraction logs."""
    torch.set_num_threads(1)  # master keeps the P-cores for the update
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = ActorCritic(**model_config)
    env_mgr = EnvironmentManager(num_envs, gamma)
    shared: RolloutBuffer | None = None

    while True:
        state_dict = conn.recv()
        if state_dict is None:
            return
        model.load_state_dict(state_dict)
        buf, completed, rollout_info = _collect_rollout(
            model, env_mgr, rollout_length, gamma, lam, device
        )
        if shared is None:
            # First rollout defines the shared storage; the master keeps the handles
            # and clones out of them each iteration.
            for f in fields(RolloutBuffer):
                getattr(buf, f.name).share_memory_()
            shared = buf
            conn.send(("buffer", shared, completed, rollout_info))
        else:
            for f in fields(RolloutBuffer):
                getattr(shared, f.name).copy_(getattr(buf, f.name))
            conn.send(("done", None, completed, rollout_info))


# =============================================================================
# PPO update
# =============================================================================


def _update_ppo(
    model,
    buffer: RolloutBuffer,
    optimizer,
    num_epochs,
    minibatch_size,
    clip_eps,
    clip_value_loss,
    coef_value,
    coef_entropy,
    max_grad_norm,
    device,
) -> dict[str, float]:
    """One PPO update over the buffer. Returns iteration-mean metrics keyed by their
    TensorBoard scalar names."""
    advantages = buffer.advantages.squeeze(-1)  # (N,)
    totals: dict[str, float] = defaultdict(float)
    n = 0

    # The decomposition's instrumentation: per-stream explained variance from
    # rollout-time values — which return stream the critic can predict (EV → 1)
    # and which carries the residual noise (typically the outcome stream).
    ev = 1.0 - (buffer.returns - buffer.values).var(dim=0) / (buffer.returns.var(dim=0) + 1e-8)
    for _ in range(num_epochs):
        # Shuffle the whole buffer once per epoch (nested-tensorclass indexing costs
        # ~25 ms per call); minibatches are then cheap contiguous slice views.
        perm = torch.randperm(len(buffer), device=device)
        x_ep = buffer.x_game_state[perm]
        mb_ep = buffer.mask_batch[perm]
        opt_ep = buffer.option_idx[perm]
        sel_ep = buffer.selection_idx[perm]
        tgt_ep = buffer.target_idx[perm]
        logp_ep = buffer.log_probs_old[perm]
        adv_ep = advantages[perm]
        ret_ep = buffer.returns[perm]
        val_ep = buffer.values[perm]
        for start in range(0, len(buffer), minibatch_size):
            rows = slice(start, start + minibatch_size)
            log_probs_new, entropies, values_new = model.evaluate_actions(
                x_ep[rows], mb_ep[rows], opt_ep[rows], sel_ep[rows], tgt_ep[rows]
            )
            log_probs_old = logp_ep[rows]
            adv = adv_ep[rows]
            returns = ret_ep[rows]
            values_old = val_ep[rows]

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
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            totals["Loss/policy"] += loss_policy.item()
            totals["Loss/value"] += loss_value.item()
            totals["Loss/entropy"] += loss_entropy.item()
            ent_mean = entropies.mean(dim=0)
            totals["Entropy/option"] += ent_mean[0].item()
            totals["Entropy/selection"] += ent_mean[1].item()
            totals["Entropy/target"] += ent_mean[2].item()
            # Schulman's approx-KL estimator; clip fraction = share of moved-off ratios
            totals["Update/approx_kl"] += (
                ((ratio - 1) - (log_probs_new - log_probs_old)).mean().item()
            )
            totals["Update/clip_fraction"] += ((ratio - 1).abs() > clip_eps).float().mean().item()
            totals["Grad/norm_preclip"] += grad_norm.item()  # pre-clip total norm (else discarded)
            n += 1
    metrics = {k: v / n for k, v in totals.items()}
    for k, name in enumerate(REWARD_STREAMS):
        metrics[f"Value/ev_{name}"] = ev[k].item()
    return metrics


# =============================================================================
# Training loop
# =============================================================================


def _save_checkpoint(path, model, optimizer, iteration, total_steps) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "total_steps": total_steps,
        },
        path,
    )


def train(
    exp_name,
    num_iterations,
    log_every,
    save_every,
    model,
    model_config,
    optimizer,
    rollout_length,
    num_epochs,
    minibatch_size,
    clip_eps,
    clip_value_loss,
    gamma,
    lam,
    coef_value,
    coef_entropy_max,
    coef_entropy_min,
    entropy_decay_steps,
    max_grad_norm,
    num_envs,
    device,
    start_iteration=0,
    total_steps=0,
    overlap_rollout=False,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)
    ckpt_path = f"experiments/{exp_name}/checkpoint.pth"
    iteration = start_iteration  # for the interrupt save, if it fires pre-loop

    # Eval runs a fixed-seed greedy battery in a persistent side process (mirrors the
    # rollout worker): weights are sent on the save cadence and the aggregate metrics
    # drained when ready, never blocking the training loop.
    eval_ctx = mp.get_context("spawn")
    eval_conn, eval_child = eval_ctx.Pipe()
    eval_proc = eval_ctx.Process(
        target=_eval_battery_worker, args=(eval_child, model_config, gamma), daemon=True
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
        worker_conn.send(model.state_dict())  # kick off the first rollout
        shared_buffer = None
    else:
        env_mgr = EnvironmentManager(num_envs, gamma)

    try:
        for iteration in range(start_iteration, num_iterations):
            # Entropy coef keyed to env steps (not iterations), linear decay to the
            # floor — consistent across resumes since it derives from total_steps.
            frac = min(1.0, total_steps / entropy_decay_steps)
            coef_entropy = coef_entropy_max + frac * (coef_entropy_min - coef_entropy_max)
            t_start = time.perf_counter()
            if overlap_rollout:
                # Wait out whatever rollout time the update didn't hide, clone the
                # shared buffer, and immediately restart the worker on fresh weights.
                kind, payload, completed, rollout_info = worker_conn.recv()
                if kind == "buffer":
                    shared_buffer = payload
                buffer = _buffer_clone(shared_buffer)
                worker_conn.send(model.state_dict())
            else:
                buffer, completed, rollout_info = _collect_rollout(
                    model, env_mgr, rollout_length, gamma, lam, device
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
                max_grad_norm,
                device,
            )
            t_update = time.perf_counter()
            # Eval battery (persistent worker): drain a finished one and, on the save
            # cadence when idle, launch a new one - both non-blocking. Logged at eval_iter.
            if eval_busy and eval_conn.poll():
                eval_iter, eval_metrics = eval_conn.recv()
                eval_busy = False
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, eval_iter)
                print(
                    f"  eval@{eval_iter}: win_rate={eval_metrics['Eval/win_rate']:.3f} "
                    f"floor={eval_metrics['Eval/avg_floor']:.2f} "
                    f"reward={eval_metrics['Eval/reward_mean']:.4f}"
                )
            if iteration % save_every == 0 and not eval_busy:
                eval_conn.send((model.state_dict(), iteration))
                eval_busy = True
            if iteration % log_every == 0:
                print(
                    f"Iter {iteration} | steps={total_steps} | "
                    f"policy={metrics['Loss/policy']:.4f} value={metrics['Loss/value']:.4f} | "
                    f"episodes={len(completed)}"
                )
                for key, value in metrics.items():
                    writer.add_scalar(key, value, iteration)
                writer.add_scalar("Entropy/coef", coef_entropy, iteration)
                writer.add_scalar("Steps/total", total_steps, iteration)
                writer.add_scalar("Time/rollout", t_rollout - t_start, iteration)
                writer.add_scalar("Time/update", t_update - t_rollout, iteration)
                writer.add_scalar("Pack/width", model.core.last_pack_width, iteration)
                for key, value in _behavioral_metrics(buffer).items():
                    writer.add_scalar(key, value, iteration)
                for key, value in rollout_info.items():
                    writer.add_scalar(key, value, iteration)
                if completed:
                    avg_r = sum(e.total_reward for e in completed) / len(completed)
                    avg_l = sum(e.length for e in completed) / len(completed)
                    writer.add_scalar("Episode/avg_reward", avg_r, iteration)
                    for k, name in enumerate(REWARD_STREAMS):
                        writer.add_scalar(
                            f"Episode/avg_reward_{name}",
                            sum(e.stream_rewards[k] for e in completed) / len(completed),
                            iteration,
                        )
                    writer.add_scalar("Episode/avg_length", avg_l, iteration)
                    writer.add_scalar("Episode/completed_count", len(completed), iteration)
                    writer.add_scalar(
                        "Episode/win_rate",
                        sum(e.won for e in completed) / len(completed),
                        iteration,
                    )
                    writer.add_scalar(
                        "Episode/avg_floor",
                        sum(e.floor for e in completed) / len(completed),
                        iteration,
                    )
            if iteration % save_every == 0:
                _save_checkpoint(ckpt_path, model, optimizer, iteration, total_steps)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        _save_checkpoint(ckpt_path, model, optimizer, iteration, total_steps)
    finally:
        if overlap_rollout and rollout_worker.is_alive():
            try:
                worker_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        if eval_proc.is_alive():
            try:
                eval_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
    writer.close()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    # nohup-backgrounded processes ignore SIGINT; route SIGTERM into the same
    # KeyboardInterrupt path so `kill <pid>` checkpoints the current iteration.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ActorCritic(**config["model"])
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])
    os.makedirs(f"experiments/{config['exp_name']}", exist_ok=True)
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    start_iteration = 0
    total_steps = 0
    ckpt_path = f"experiments/{config['exp_name']}/checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        total_steps = ckpt["total_steps"]
        # Offset the stream seeds so a resume doesn't replay the run's env-seed sequence
        random.seed(seed + start_iteration)
        torch.manual_seed(seed + start_iteration)
        print(f"Resuming from {ckpt_path}: iteration {start_iteration}, {total_steps} steps")

    print(f"Starting training: {config['exp_name']}")
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
        max_grad_norm=config["max_grad_norm"],
        num_envs=config["num_envs"],
        device=torch.device("cpu"),
        start_iteration=start_iteration,
        total_steps=total_steps,
        overlap_rollout=bool(config.get("overlap_rollout", False)),
    )
