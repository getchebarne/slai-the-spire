from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.algorithms.actor_critic.distributed import average_gradients
from src.rl.algorithms.actor_critic.distributed import mean_over_ranks
from src.rl.models import ActorCritic
from src.rl.packed import take_states
from src.rl.reward import RewardStream
from src.rl.types import X_Rollout
from src.rl.types import X_TrainingBatch


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
    model: ActorCritic, x_rollout: X_Rollout, minibatch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the rollout's log-probs and values under the Learner's current weights.

    The model runs a minibatch of states at a time.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model.
    x_rollout : X_Rollout
        The rollout: T steps of E games.
    minibatch_size : int
        The most states per forward.

    Returns
    -------
    x_log_probs_prox : torch.Tensor
        (N,). Each state's recorded action's log-prob, summed over the levels.
    x_values_prox : torch.Tensor
        (T, E, K). Each state's values.
    x_values_final : torch.Tensor
        (E, K). Each game's final value.
    """
    T, E, K = x_rollout.x_rewards.shape
    log_probs, values = [], []
    x_order = torch.arange(T * E, device=x_rollout.x_actions.device)
    starts = range(0, T * E, minibatch_size)

    # Each minibatch's rows in each block, summed on the CPU before the first one runs
    x_count_totals = [
        x_rollout.x_count_cpu[start : start + minibatch_size].sum(0) for start in starts
    ]
    with torch.no_grad():
        for start, x_count_total in zip(starts, x_count_totals):
            x_states = x_order[start : start + minibatch_size]
            x_log_probs, _, _, x_values = model.evaluate_actions(
                take_states(x_rollout.x_game_state, x_states, x_count_total),
                x_rollout.x_actions[x_states],
            )
            log_probs.append(x_log_probs)
            values.append(x_values)

        # Bootstrap value V(s_T) per env (a fresh env's value if it reset on the last step;
        # GAE's done-masking ensures that only contributes to non-terminal timesteps).
        x_values_final = model(x_rollout.x_game_state_final)
    return torch.cat(log_probs), torch.cat(values).reshape(T, E, K), x_values_final


def prepare_batch(
    model: ActorCritic, x_rollout: X_Rollout, gamma: float, ppo_config: PPOConfig
) -> X_TrainingBatch:
    """Prepare the update's batch: GAE, and the anchor π_prox the clip and the KL measure from.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model; its current weights give π_prox and the values.
    x_rollout : X_Rollout
        The collected rollout.
    gamma : float
        The discount.
    ppo_config : PPOConfig
        The PPO settings, for GAE's λ and the minibatch size.

    Returns
    -------
    X_TrainingBatch
        The rollout's states, actions and behavior log-probs, with π_prox's log-probs and
        values, the returns and the normalized advantages.
    """
    T, E, K = x_rollout.x_rewards.shape
    x_log_probs_prox, x_values_prox, x_values_final = _recompute(
        model, x_rollout, ppo_config.minibatch_size
    )

    x_returns, x_advantages = _compute_gae(
        x_rollout.x_rewards,
        x_values_prox,
        x_rollout.x_dones.unsqueeze(-1),
        x_values_final,
        gamma,
        ppo_config.lam,
    )

    # Policy advantage: per-stream advantages summed (≡ single-critic GAE on the
    # summed reward, by linearity)
    x_advantages = x_advantages.sum(-1)

    # Normalize advantages over the batch
    x_advantages = (x_advantages - x_advantages.mean()) / (x_advantages.std() + 1e-8)
    return X_TrainingBatch(
        x_game_state=x_rollout.x_game_state,
        x_count_cpu=x_rollout.x_count_cpu,
        x_actions=x_rollout.x_actions,
        x_log_probs_behavior=x_rollout.x_log_probs_behavior,
        x_log_probs_prox=x_log_probs_prox,
        x_values_prox=x_values_prox.reshape(-1, K),
        x_returns=x_returns.reshape(-1, K),
        x_advantages=x_advantages.reshape(-1, 1),
    )


def _minibatch_metrics(
    loss_policy: torch.Tensor,
    loss_value: torch.Tensor,
    loss_entropy: torch.Tensor,
    entropies: torch.Tensor,
    loss_kl_uniform: torch.Tensor,
    kl_uniform: torch.Tensor,
    ratio: torch.Tensor,
    log_probs_new: torch.Tensor,
    log_probs_prox: torch.Tensor,
    grad_norm: torch.Tensor,
    clip_eps: float,
) -> dict[str, float]:
    """Return the per-minibatch TensorBoard scalars (minibatch means).

    Schulman's approx-KL estimator; clip fraction = share of moved-off ratios; grad norm is the
    pre-clip total (otherwise discarded by clip_grad_norm_).

    Parameters
    ----------
    loss_policy : torch.Tensor
        (). The policy loss.
    loss_value : torch.Tensor
        (). The value loss.
    loss_entropy : torch.Tensor
        (). The entropy loss.
    entropies : torch.Tensor
        (B, 3). The entropy of each of the minibatch's B states at each level.
    loss_kl_uniform : torch.Tensor
        (). The uniform-KL loss.
    kl_uniform : torch.Tensor
        (B, 3). Each state's KL(uniform‖π) at each level.
    ratio : torch.Tensor
        (B,). Each recorded action's probability ratio, the current weights' over π_prox's.
    log_probs_new : torch.Tensor
        (B,). The recorded actions' log-probs under the current weights.
    log_probs_prox : torch.Tensor
        (B,). The recorded actions' log-probs under π_prox.
    grad_norm : torch.Tensor
        (). The gradients' total norm before clipping.
    clip_eps : float
        The clip range.

    Returns
    -------
    dict[str, float]
        Each scalar under its TensorBoard name.
    """
    ent_mean = entropies.mean(dim=0)
    kl_uniform_mean = kl_uniform.mean(dim=0)
    metrics = {
        "loss/policy": loss_policy,
        "loss/value": loss_value,
        "loss/entropy": loss_entropy,
        "loss/kl_uniform": loss_kl_uniform,
        "entropy/L1": ent_mean[0],
        "entropy/L2": ent_mean[1],
        "entropy/L3": ent_mean[2],
        "kl_uniform/L1": kl_uniform_mean[0],
        "kl_uniform/L2": kl_uniform_mean[1],
        "kl_uniform/L3": kl_uniform_mean[2],
        "update/approx_kl": ((ratio - 1) - (log_probs_new - log_probs_prox)).mean(),
        "update/clip_fraction": ((ratio - 1).abs() > clip_eps).float().mean(),
        "grad/norm_preclip": grad_norm,
    }

    # One read for all of them: each .item() would make the CPU wait for the GPU
    values = torch.stack([x.detach() for x in metrics.values()]).tolist()
    return dict(zip(metrics, values))


def update_ppo(
    model: ActorCritic,
    batch: X_TrainingBatch,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Run one PPO update over the batch.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model, updated in place.
    batch : X_TrainingBatch
        The update's batch.
    optimizer : torch.optim.Optimizer
        The model's optimizer.
    ppo_config : PPOConfig
        The PPO settings.
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
    records: list[dict[str, float]] = []

    # Per-stream explained variance of the values the Learner recomputed before the update:
    # which return stream the critic predicts (EV → 1) and which carries the residual noise
    ev = 1.0 - (batch.x_returns - batch.x_values_prox).var(dim=0) / (
        batch.x_returns.var(dim=0) + 1e-8
    )

    # How far the anchor sits from the weights that played: the off-policy weight
    # π_prox / π_behavior, its truncation rate, and KL(π_behavior ‖ π_prox)
    log_weight = batch.x_log_probs_prox - batch.x_log_probs_behavior
    weight = log_weight.exp()
    offpolicy = {
        "offpolicy/is_weight_mean": weight.clamp(max=ppo_config.is_clip).mean().item(),
        "offpolicy/is_weight_truncated": (weight > ppo_config.is_clip).float().mean().item(),
        "offpolicy/kl_behavior_prox": (weight - 1 - log_weight).mean().item(),
    }

    # Every epoch's shuffle, planned before the first minibatch runs: the minibatches' states, sent
    # to the device in one copy, and their rows in each block, summed on the CPU. No CPU work then
    # sits between a minibatch's KL read and the next minibatch's launch
    batch_size = len(batch.x_actions)
    starts = range(0, batch_size, ppo_config.minibatch_size)
    x_order = torch.stack(
        [torch.randperm(batch_size, generator=generator) for _ in range(ppo_config.num_epochs)]
    )
    x_count_totals = [
        [
            batch.x_count_cpu[x_order_epoch[start : start + ppo_config.minibatch_size]].sum(0)
            for start in starts
        ]
        for x_order_epoch in x_order
    ]
    x_order = (x_order.pin_memory() if device.type == "cuda" else x_order).to(
        device, non_blocking=True
    )

    epochs_run = 0.0
    for epoch in range(ppo_config.num_epochs):
        epoch_start = len(records)
        stopped = False
        for start, x_count_total in zip(starts, x_count_totals[epoch]):
            x_states = x_order[epoch, start : start + ppo_config.minibatch_size]
            log_probs_new, entropies, kl_uniform, values_new = model.evaluate_actions(
                take_states(batch.x_game_state, x_states, x_count_total),
                batch.x_actions[x_states],
            )
            log_probs_prox = batch.x_log_probs_prox[x_states]
            adv = batch.x_advantages[x_states].squeeze(-1)
            returns = batch.x_returns[x_states]
            values_prox = batch.x_values_prox[x_states]

            # Decoupled PPO (Hilton et al. 2021): clip against π_prox, and weight each
            # sample by π_prox / π_behavior, capped, for the weights that played
            ratio = torch.exp(log_probs_new - log_probs_prox)
            weight = torch.exp(log_probs_prox - batch.x_log_probs_behavior[x_states]).clamp(
                max=ppo_config.is_clip
            )
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * adv
            loss_policy = -torch.mean(weight * torch.min(surr1, surr2))

            # Value loss: summed over streams, mean over the batch, so each stream's gradient
            # keeps the single-critic scale (a mean over K would divide coef_value by K)
            if ppo_config.clip_value_loss:
                values_clipped = values_prox + torch.clamp(
                    values_new - values_prox, -ppo_config.clip_eps, ppo_config.clip_eps
                )
                lv_unclipped = torch.pow(values_new - returns, 2)
                lv_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.max(lv_unclipped, lv_clipped).sum(dim=-1).mean()
            else:
                # 0.5·MSE, the clipped branch's canonical PPO scale: without the 0.5,
                # toggling clip_value_loss would double the effective critic weight
                loss_value = 0.5 * F.mse_loss(values_new, returns, reduction="none").sum(-1).mean()

            loss_entropy = -torch.mean(entropies.sum(dim=-1))

            # Uniform-KL barrier, summed per head like the entropy term; each head's KL falls
            # only by lifting its own tail probs. Never decay it: it keeps rare actions reachable
            loss_kl_uniform = torch.mean(kl_uniform.sum(dim=-1))
            loss = (
                loss_policy
                + ppo_config.coef_value * loss_value
                + ppo_config.coef_entropy * loss_entropy
                + ppo_config.coef_kl_uniform * loss_kl_uniform
            )

            optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), ppo_config.max_grad_norm)
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
                    log_probs_prox,
                    grad_norm,
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
        mb_total = (batch_size + ppo_config.minibatch_size - 1) // ppo_config.minibatch_size
        epochs_run += (len(records) - epoch_start) / mb_total
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
        metrics[f"Value/ev_{name}"] = ev[k].item()
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
