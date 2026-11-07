from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.algorithms.actor_critic.distributed import average_gradients
from src.rl.algorithms.actor_critic.distributed import mean_over_ranks
from src.rl.models import ActorCritic
from src.rl.packed import build_batch_layout
from src.rl.packed import plan_minibatches
from src.rl.packed import slice_states
from src.rl.packed import take_states
from src.rl.reward import RewardStream
from src.rl.types import BatchLayout
from src.rl.types import Rollout


@dataclass
class PPOConfig:
    lam: float
    num_epochs: int
    minibatch_size: int
    clip_eps: float
    clip_value_loss: bool
    coef_value: float
    coef_entropy: float
    coef_kl_uniform: float
    is_clip: float
    max_grad_norm: float
    target_kl: float | None


def _compute_gae(
    x_rewards: torch.Tensor,
    x_values: torch.Tensor,
    x_dones: torch.Tensor,
    x_values_final: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over E parallel games, one recursion per reward stream.

    A done zeroes the next value and the carried advantage, so nothing leaks across an episode
    boundary.

    Parameters
    ----------
    x_rewards : torch.Tensor
        (T, E, K) float32. Each step's rewards, one per reward stream.
    x_values : torch.Tensor
        (T, E, K). Each state's values.
    x_dones : torch.Tensor
        (T, E, 1) float32. Whether each step ended its game's episode: 1.0 if so, else 0.0.
    x_values_final : torch.Tensor
        (E, K). Each game's value after the last step.
    gamma : float
        The discount.
    lam : float
        GAE's λ.

    Returns
    -------
    x_returns : torch.Tensor
        (T, E, K). Each step's advantages plus its values.
    x_advantages : torch.Tensor
        (T, E, K). Each step's advantages.
    """
    T = x_rewards.shape[0]
    x_advantages = torch.zeros_like(x_rewards)
    x_gae = torch.zeros_like(x_values_final)
    for t in reversed(range(T)):
        x_value_next = x_values_final if t == T - 1 else x_values[t + 1]
        x_non_terminal = 1.0 - x_dones[t]
        x_delta = x_rewards[t] + gamma * x_value_next * x_non_terminal - x_values[t]
        x_gae = x_delta + gamma * lam * x_non_terminal * x_gae
        x_advantages[t] = x_gae
    return x_advantages + x_values, x_advantages


def _recompute(
    model: ActorCritic, rollout: Rollout, minibatch_size: int, layout: BatchLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the rollout's log-probs and values under the Learner's current weights.

    The model runs a minibatch of states at a time.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model.
    rollout : Rollout
        The rollout: T steps of E games.
    minibatch_size : int
        The most states per forward.
    layout : BatchLayout
        The rollout's cached boundaries, shared with the subsequent shuffled update.

    Returns
    -------
    x_log_probs_prox : torch.Tensor
        (N,). Each state's recorded action's log-prob, summed over the levels.
    x_values_prox : torch.Tensor
        (T, E, K). Each state's values.
    x_values_final : torch.Tensor
        (E, K). Each game's final value.
    """
    T, E, K = rollout.x_rewards.shape
    log_probs, values = [], []
    with torch.no_grad():
        for start in range(0, T * E, minibatch_size):
            stop = start + minibatch_size
            x_log_probs, _, _, x_values = model.evaluate_actions(
                slice_states(rollout.states, start, stop, layout),
                rollout.x_actions[start:stop],
            )
            log_probs.append(x_log_probs)
            values.append(x_values)

        # Bootstrap value V(s_T) per env (a fresh env's value if it reset on the last step;
        # GAE's done-masking ensures that only contributes to non-terminal timesteps).
        x_values_final = model(rollout.final_states)
    return torch.cat(log_probs), torch.cat(values).reshape(T, E, K), x_values_final


def _minibatch_metrics(
    x_loss_policy: torch.Tensor,
    x_loss_value: torch.Tensor,
    x_loss_entropy: torch.Tensor,
    x_entropies: torch.Tensor,
    x_loss_kl_uniform: torch.Tensor,
    x_kl_uniform: torch.Tensor,
    x_ratio: torch.Tensor,
    x_log_probs_new: torch.Tensor,
    x_log_probs_prox: torch.Tensor,
    x_grad_norm: torch.Tensor,
    clip_eps: float,
) -> dict[str, float]:
    """Return the per-minibatch TensorBoard scalars (minibatch means).

    Schulman's approx-KL estimator; clip fraction = share of moved-off ratios; grad norm is the
    pre-clip total (otherwise discarded by clip_grad_norm_).

    Parameters
    ----------
    x_loss_policy : torch.Tensor
        (). The policy loss.
    x_loss_value : torch.Tensor
        (). The value loss.
    x_loss_entropy : torch.Tensor
        (). The entropy loss.
    x_entropies : torch.Tensor
        (B, 3). The entropy of each of the minibatch's B states at each level.
    x_loss_kl_uniform : torch.Tensor
        (). The uniform-KL loss.
    x_kl_uniform : torch.Tensor
        (B, 3). Each state's KL(uniform‖π) at each level.
    x_ratio : torch.Tensor
        (B,). Each recorded action's probability ratio, the current weights' over π_prox's.
    x_log_probs_new : torch.Tensor
        (B,). The recorded actions' log-probs under the current weights.
    x_log_probs_prox : torch.Tensor
        (B,). The recorded actions' log-probs under π_prox.
    x_grad_norm : torch.Tensor
        (). The gradients' total norm before clipping.
    clip_eps : float
        The clip range.

    Returns
    -------
    dict[str, float]
        Each scalar under its TensorBoard name.
    """
    x_ent_mean = x_entropies.mean(dim=0)
    x_kl_uniform_mean = x_kl_uniform.mean(dim=0)
    metrics = {
        "loss/policy": x_loss_policy,
        "loss/value": x_loss_value,
        "loss/entropy": x_loss_entropy,
        "loss/kl_uniform": x_loss_kl_uniform,
        "entropy/L1": x_ent_mean[0],
        "entropy/L2": x_ent_mean[1],
        "entropy/L3": x_ent_mean[2],
        "kl_uniform/L1": x_kl_uniform_mean[0],
        "kl_uniform/L2": x_kl_uniform_mean[1],
        "kl_uniform/L3": x_kl_uniform_mean[2],
        "update/approx_kl": ((x_ratio - 1) - (x_log_probs_new - x_log_probs_prox)).mean(),
        "update/clip_fraction": ((x_ratio - 1).abs() > clip_eps).float().mean(),
        "grad/norm_preclip": x_grad_norm,
    }

    # One read for all of them: each .item() would make the CPU wait for the GPU
    values = torch.stack([x_metric.detach() for x_metric in metrics.values()]).tolist()
    return dict(zip(metrics, values))


def update_ppo(
    model: ActorCritic,
    rollout: Rollout,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    gamma: float,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Recompute the rollout's anchor policy and GAE, then run one PPO update.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model, updated in place.
    rollout : Rollout
        The collected states, actions and rewards.
    optimizer : torch.optim.Optimizer
        The model's optimizer.
    ppo_config : PPOConfig
        The PPO settings.
    gamma : float
        The discount.
    device : torch.device
        The Learner's device.
    generator : torch.Generator
        The CPU generator the epochs' shuffles draw from.

    Returns
    -------
    dict[str, float]
        The update's metrics, keyed by their TensorBoard scalar names: each per-minibatch metric
        averaged over the minibatches run, then `update/epochs_run`, the `Value/ev_*` explained
        variances and the `offpolicy/*` scalars, each computed once per update.
    """
    K = rollout.x_rewards.shape[-1]
    layout = build_batch_layout(rollout.states, rollout.x_counts_cpu)
    x_log_probs_prox, x_values_prox, x_values_final = _recompute(
        model, rollout, ppo_config.minibatch_size, layout
    )
    x_returns, x_advantages = _compute_gae(
        rollout.x_rewards,
        x_values_prox,
        rollout.x_dones.unsqueeze(-1),
        x_values_final,
        gamma,
        ppo_config.lam,
    )

    # Policy advantage: per-stream advantages summed (equivalent to one summed-reward critic).
    x_advantages = x_advantages.sum(-1)
    x_advantages = (x_advantages - x_advantages.mean()) / (x_advantages.std() + 1e-8)
    x_values_prox = x_values_prox.reshape(-1, K)
    x_returns = x_returns.reshape(-1, K)
    x_advantages = x_advantages.reshape(-1, 1)

    records: list[dict[str, float]] = []

    # Per-stream explained variance of the values the Learner recomputed before the update:
    # which return stream the critic predicts (EV → 1) and which carries the residual noise
    x_ev = 1.0 - (x_returns - x_values_prox).var(dim=0) / (x_returns.var(dim=0) + 1e-8)

    # How far the anchor sits from the weights that played: the off-policy weight
    # π_prox / π_behavior, its truncation rate, and KL(π_behavior ‖ π_prox)
    x_log_weight = x_log_probs_prox - rollout.x_log_probs_behavior
    x_weight = x_log_weight.exp()
    offpolicy = {
        "offpolicy/is_weight_mean": x_weight.clamp(max=ppo_config.is_clip).mean().item(),
        "offpolicy/is_weight_truncated": (x_weight > ppo_config.is_clip).float().mean().item(),
        "offpolicy/kl_behavior_prox": (x_weight - 1 - x_log_weight).mean().item(),
    }

    # All shuffles and CPU output sizes are planned before the first minibatch launches.
    plans = plan_minibatches(
        rollout.x_counts_cpu, ppo_config.minibatch_size, ppo_config.num_epochs, device, generator
    )

    epochs_run = 0.0
    for epoch_plans in plans:
        epoch_start = len(records)
        stopped = False
        for plan in epoch_plans:
            x_states = plan.x_state_ids
            x_log_probs_new, x_entropies, x_kl_uniform, x_values_new = model.evaluate_actions(
                take_states(rollout.states, plan, layout),
                rollout.x_actions[x_states],
            )
            x_log_probs_prox_batch = x_log_probs_prox[x_states]
            x_adv = x_advantages[x_states].squeeze(-1)
            x_returns_batch = x_returns[x_states]
            x_values_prox_batch = x_values_prox[x_states]

            # Decoupled PPO (Hilton et al. 2021): clip against π_prox, and weight each
            # sample by π_prox / π_behavior, capped, for the weights that played
            x_ratio = torch.exp(x_log_probs_new - x_log_probs_prox_batch)
            x_weight = torch.exp(
                x_log_probs_prox_batch - rollout.x_log_probs_behavior[x_states]
            ).clamp(max=ppo_config.is_clip)
            x_surr1 = x_ratio * x_adv
            x_surr2 = (
                torch.clamp(x_ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * x_adv
            )
            x_loss_policy = -torch.mean(x_weight * torch.min(x_surr1, x_surr2))

            # Value loss: summed over streams, mean over the batch, so each stream's gradient
            # keeps the single-critic scale (a mean over K would divide coef_value by K)
            if ppo_config.clip_value_loss:
                x_values_clipped = x_values_prox_batch + torch.clamp(
                    x_values_new - x_values_prox_batch, -ppo_config.clip_eps, ppo_config.clip_eps
                )
                x_lv_unclipped = torch.pow(x_values_new - x_returns_batch, 2)
                x_lv_clipped = torch.pow(x_values_clipped - x_returns_batch, 2)
                x_loss_value = 0.5 * torch.max(x_lv_unclipped, x_lv_clipped).sum(dim=-1).mean()
            else:
                # 0.5·MSE, the clipped branch's canonical PPO scale: without the 0.5,
                # toggling clip_value_loss would double the effective critic weight
                x_loss_value = (
                    0.5
                    * F.mse_loss(x_values_new, x_returns_batch, reduction="none").sum(-1).mean()
                )

            x_loss_entropy = -torch.mean(x_entropies.sum(dim=-1))

            # Uniform-KL barrier, summed per head like the entropy term; each head's KL falls
            # only by lifting its own tail probs. Never decay it: it keeps rare actions reachable
            x_loss_kl_uniform = torch.mean(x_kl_uniform.sum(dim=-1))
            x_loss = (
                x_loss_policy
                + ppo_config.coef_value * x_loss_value
                + ppo_config.coef_entropy * x_loss_entropy
                + ppo_config.coef_kl_uniform * x_loss_kl_uniform
            )

            optimizer.zero_grad()
            x_loss.backward()
            average_gradients(model)
            x_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), ppo_config.max_grad_norm)
            optimizer.step()

            records.append(
                _minibatch_metrics(
                    x_loss_policy,
                    x_loss_value,
                    x_loss_entropy,
                    x_entropies,
                    x_loss_kl_uniform,
                    x_kl_uniform,
                    x_ratio,
                    x_log_probs_new,
                    x_log_probs_prox_batch,
                    x_grad_norm,
                    ppo_config.clip_eps,
                )
            )

            # Every rank stops on the same KL, or the others would wait in the next all-reduce
            records[-1]["update/approx_kl"] = mean_over_ranks(
                records[-1]["update/approx_kl"], device
            )

            # Minibatch trust region: an epoch is many unchecked optimizer steps. 3x, not the
            # epoch stop's 1.5x: a single minibatch's KL estimate is noisy
            if (
                ppo_config.target_kl is not None
                and records[-1]["update/approx_kl"] > 3.0 * ppo_config.target_kl
            ):
                stopped = True
                break
        epochs_run += (len(records) - epoch_start) / len(epoch_plans)
        if stopped:
            break

        # Target-KL early stop: the clip bounds each sample's incentive, not the drift
        # across epochs on a frozen log_probs_prox, so stop once it leaves the trust region
        if ppo_config.target_kl is not None:
            epoch_kl = sum(r["update/approx_kl"] for r in records[epoch_start:]) / max(
                1, len(records) - epoch_start
            )
            if epoch_kl > 1.5 * ppo_config.target_kl:
                break
    metrics = {key: sum(r[key] for r in records) / len(records) for key in records[0]}
    metrics["update/epochs_run"] = float(epochs_run)
    for k, name in enumerate(RewardStream):
        metrics[f"Value/ev_{name}"] = x_ev[k].item()
    return metrics | offpolicy


def adapt_lr_scale(lr_scale: float, approx_kl: float, target_kl: float) -> float:
    """Return the next adaptive-KL lr scale.

    Parameters
    ----------
    lr_scale : float
        The current lr scale.
    approx_kl : float
        The last update's approx KL.
    target_kl : float
        The target KL.

    Returns
    -------
    float
        `lr_scale` divided by 1.5 when `approx_kl` is above twice `target_kl`, down to 1/64;
        multiplied by 1.5 when below half of it, up to 1.0; unchanged otherwise.
    """
    if approx_kl > 2.0 * target_kl:
        return max(lr_scale / 1.5, 1.0 / 64.0)
    if approx_kl < 0.5 * target_kl:
        return min(lr_scale * 1.5, 1.0)
    return lr_scale
