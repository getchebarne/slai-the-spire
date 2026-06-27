import multiprocessing as mp
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from types import FrameType

import numpy as np
import slai
import torch
import torch.multiprocessing  # noqa: F401 — registers tensor reductions (shm handles over pipes)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.masks import build_masks
from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.encoding.state import encode_batch_game_state
from src.rl.models import ActorCritic
from src.rl.reward import REWARD_STREAMS
from src.rl.reward import compute_reward
from src.rl.types import Level
from src.rl.types import RolloutBuffer
from src.rl.utils import action_from_actiontype
from src.rl.utils import init_optimizer
from src.rl.utils import load_config
from src.rl.utils import shuffle_rollout_buffer
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import aggregate_episodes
from src.rl.algorithms.actor_critic.evals import eval_battery_worker


@dataclass
class EnvPool:
    envs: list[slai.GameEnv]
    obs: list[slai.GameState]
    ep_rewards: list[np.ndarray]  # (K,) per env, accumulated across the episode
    ep_lengths: list[int]
    completed: list[EpisodeStats]
    gamma: float


def _make_env() -> tuple[slai.GameEnv, slai.GameState]:
    env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))
    return env, obs


def make_env_pool(num_envs: int, gamma: float) -> EnvPool:
    # Episode stats live on the pool (not in _collect_rollout) so episodes spanning
    # rollout boundaries report true totals.
    envs, obs = [], []
    for _ in range(num_envs):
        env, ob = _make_env()
        envs.append(env)
        obs.append(ob)
    return EnvPool(
        envs=envs,
        obs=obs,
        ep_rewards=[np.zeros(len(REWARD_STREAMS)) for _ in range(num_envs)],
        ep_lengths=[0] * num_envs,
        completed=[],
        gamma=gamma,
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
                floor=nxt.map.y_current or 0,
            )
        )
        pool.ep_rewards[env_idx] = np.zeros(len(REWARD_STREAMS))
        pool.ep_lengths[env_idx] = 0
        pool.envs[env_idx], nxt = _make_env()
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

            # Decode actions on CPU once (one device->host transfer, not per-env), then step.
            out_cpu = out.cpu()
            for i in range(E):
                reward, done = step_env(pool, i, action_from_actiontype(out_cpu, i))
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
    rollout_info = {
        "rollout/terminations": dones.sum().item(),
        "rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
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


def _rollout_worker(
    conn: Connection,
    model_config: dict,
    num_envs: int,
    rollout_length: int,
    gamma: float,
    lam: float,
    seed: int,
) -> None:
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
    pool = make_env_pool(num_envs, gamma)
    shared: RolloutBuffer | None = None

    while True:
        # Receive and load current model weights
        state_dict = conn.recv()
        if state_dict is None:
            return

        model.load_state_dict(state_dict)

        # Collect rollout
        buf, completed, rollout_info = _collect_rollout(
            model, pool, rollout_length, gamma, lam, device
        )
        if shared is None:
            # First rollout defines the shared storage; the master keeps the handles
            # and clones out of them each iteration.
            buf.share_memory_()
            shared = buf
            conn.send(("buffer", shared, completed, rollout_info))
        else:
            shared.copy_(buf)
            conn.send(("done", None, completed, rollout_info))


def _minibatch_metrics(
    loss_policy: torch.Tensor,
    loss_value: torch.Tensor,
    loss_entropy: torch.Tensor,
    entropies: torch.Tensor,
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
    return {
        "loss/policy": loss_policy.item(),
        "loss/value": loss_value.item(),
        "loss/entropy": loss_entropy.item(),
        "entropy/L1": ent_mean[0].item(),
        "entropy/L2": ent_mean[1].item(),
        "entropy/L3": ent_mean[2].item(),
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
    max_grad_norm: float,
    device: torch.device,
) -> dict[str, float]:
    """One PPO update over the buffer. Returns iteration-mean metrics keyed by their
    TensorBoard scalar names."""
    records: list[dict[str, float]] = []

    # The decomposition's instrumentation: per-stream explained variance from
    # rollout-time values — which return stream the critic can predict (EV → 1)
    # and which carries the residual noise (typically the outcome stream).
    ev = 1.0 - (buffer.returns - buffer.values).var(dim=0) / (buffer.returns.var(dim=0) + 1e-8)
    for _ in range(num_epochs):
        ep = shuffle_rollout_buffer(buffer, device)
        for start in range(0, len(buffer), minibatch_size):
            rows = slice(start, start + minibatch_size)
            log_probs_new, entropies, values_new = model.evaluate_actions(
                ep.game_state[rows],
                ep.mask_batch[rows],
                ep.idx_at[rows],
                ep.idx_l1[rows],
                ep.idx_l2[rows],
            )
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
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

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
                    ratio,
                    log_probs_new,
                    log_probs_old,
                    grad_norm,
                    clip_eps,
                )
            )
    metrics = {key: sum(r[key] for r in records) / len(records) for key in records[0]}
    for k, name in enumerate(REWARD_STREAMS):
        metrics[f"Value/ev_{name}"] = ev[k].item()
    return metrics


def _save_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    total_steps: int,
) -> None:
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
    max_grad_norm: float,
    num_envs: int,
    device: torch.device,
    start_iteration: int = 0,
    total_steps: int = 0,
    overlap_rollout: bool = False,
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
        worker_conn.send(model.state_dict())  # kick off the first rollout
        shared_buffer = None
    else:
        pool = make_env_pool(num_envs, gamma)

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
                buffer = shared_buffer.clone()
                worker_conn.send(model.state_dict())
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
                    f"policy={metrics['loss/policy']:.4f} value={metrics['loss/value']:.4f} | "
                    f"episodes={len(completed)}"
                )
                for key, value in metrics.items():
                    writer.add_scalar(key, value, iteration)
                writer.add_scalar("entropy/coef", coef_entropy, iteration)
                writer.add_scalar("Steps/total", total_steps, iteration)
                writer.add_scalar("Time/rollout", t_rollout - t_start, iteration)
                writer.add_scalar("Time/update", t_update - t_rollout, iteration)
                for key, value in rollout_info.items():
                    writer.add_scalar(key, value, iteration)
                if completed:
                    for key, value in aggregate_episodes(completed).items():
                        writer.add_scalar(f"Episode/{key}", value, iteration)
                    writer.add_scalar("Episode/completed_count", len(completed), iteration)
                    for k, name in enumerate(REWARD_STREAMS):
                        writer.add_scalar(
                            f"Episode/reward_avg_{name}",
                            sum(e.stream_rewards[k] for e in completed) / len(completed),
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


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def _load_checkpoint(
    ckpt_path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> tuple[int, int]:
    """Resume model/optimizer/RNG from `ckpt_path` if it exists; returns
    (start_iteration, total_steps), or (0, 0) for a fresh run."""
    if not os.path.exists(ckpt_path):
        return 0, 0

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iteration = ckpt["iteration"] + 1
    total_steps = ckpt["total_steps"]
    # Offset the stream seeds so a resume doesn't replay the run's env-seed sequence
    random.seed(seed + start_iteration)
    torch.manual_seed(seed + start_iteration)
    print(f"Resuming from {ckpt_path}: iteration {start_iteration}, {total_steps} steps")
    return start_iteration, total_steps


if __name__ == "__main__":
    # nohup-backgrounded processes ignore SIGINT; route SIGTERM into the same
    # KeyboardInterrupt path so `kill <pid>` checkpoints the current iteration.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    config_path = "src/rl/algorithms/actor_critic/config.yml"
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
    start_iteration, total_steps = _load_checkpoint(ckpt_path, model, optimizer, seed)

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
        max_grad_norm=config["max_grad_norm"],
        num_envs=config["num_envs"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        start_iteration=start_iteration,
        total_steps=total_steps,
        overlap_rollout=bool(config.get("overlap_rollout", False)),
    )
