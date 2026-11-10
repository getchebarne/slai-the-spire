from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.algorithms.actor_critic.distributed import average_gradients
from src.rl.algorithms.actor_critic.distributed import mean_over_ranks
from src.rl.algorithms.actor_critic.metrics import MetricKey
from src.rl.batches import build_state_batch_layout
from src.rl.batches import slice_state_batch
from src.rl.batches import take_minibatch_states
from src.rl.models import ActorCritic
from src.rl.reward import RewardStream
from src.rl.types import MinibatchPlan
from src.rl.types import Rollout
from src.rl.types import StateBatchLayout


@dataclass
class PPOConfig:
    lam: float
    num_epochs: int
    minibatch_size: int
    clip_eps: float
    coef_value: float
    coef_entropy: float
    coef_kl_uniform: float
    is_clip: float
    max_grad_norm: float
    target_kl: float | None


@dataclass
class _Targets:
    """What an update's minibatches train against: one row per state, fixed for the update."""

    x_log_probs_prox: torch.Tensor  # (num_states,) the recorded actions' log-probs under π_prox
    x_returns: torch.Tensor  # (num_states, num_reward_streams)
    x_advantages: torch.Tensor  # (num_states,) summed over the reward streams, normalized
    x_weights_capped: torch.Tensor  # (num_states,) π_prox / π_behavior, capped


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
        averaged over the minibatches run, then `update/epochs_run`, the `value/ev_*` explained
        variances and the `offpolicy/*` scalars, each computed once per update.
    """
    num_reward_streams = rollout.x_rewards.shape[-1]
    state_batch_layout = build_state_batch_layout(rollout.x_counts_cpu, device)

    # Calculate the rollout's log-probabilities and values under the current weights
    x_log_probs_prox, x_value_prox, x_value_final = _evaluate_rollout(
        model, rollout, ppo_config.minibatch_size, state_batch_layout
    )

    # Compute GAE returns and advantages
    x_returns, x_advantages = _compute_gae(
        rollout.x_rewards,
        x_value_prox,
        rollout.x_dones.unsqueeze(-1),
        x_value_final,
        gamma,
        ppo_config.lam,
    )

    # Sum per-stream advantages (advantage is linear) and normalize them over the batch
    x_advantages = x_advantages.sum(-1)
    x_advantages = (x_advantages - x_advantages.mean()) / (x_advantages.std() + 1e-8)

    # One row per state from here on
    x_log_weights = x_log_probs_prox - rollout.x_log_probs_behavior
    x_weights = x_log_weights.exp()
    targets = _Targets(
        x_log_probs_prox=x_log_probs_prox,
        x_returns=x_returns.reshape(-1, num_reward_streams),
        x_advantages=x_advantages.reshape(-1),
        x_weights_capped=x_weights.clamp(max=ppo_config.is_clip),
    )

    # All shuffles and CPU output sizes are planned before the first minibatch launches.
    minibatch_plans = _plan_minibatches(
        rollout.x_counts_cpu, ppo_config.minibatch_size, ppo_config.num_epochs, device, generator
    )
    num_minibatches = len(minibatch_plans) // ppo_config.num_epochs

    batch_metrics = []
    for minibatch_plan in minibatch_plans:
        metrics = _update_minibatch(
            model,
            optimizer,
            rollout,
            state_batch_layout,
            minibatch_plan,
            targets,
            ppo_config,
            device,
        )
        batch_metrics.append(metrics)

        # Trust-region early stop check
        if ppo_config.target_kl is not None and _left_trust_region(
            batch_metrics, num_minibatches, ppo_config.target_kl
        ):
            break

    # Each per-minibatch metric averaged over the minibatches run, then the once-per-update ones
    return _average_minibatch_metrics(batch_metrics, num_minibatches) | _anchor_metrics(
        targets.x_returns, x_value_prox, x_log_weights, x_weights, ppo_config.is_clip
    )


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


def _evaluate_rollout(
    model: ActorCritic, rollout: Rollout, minibatch_size: int, state_batch_layout: StateBatchLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the rollout's log-probs and values under the Learner's current weights.

    The model runs a minibatch of states at a time.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model.
    rollout : Rollout
        The rollout: `rollout_length` steps of `num_envs` games.
    minibatch_size : int
        The most states per forward.
    state_batch_layout : StateBatchLayout
        The rollout's cached offsets, shared with the subsequent shuffled update.

    Returns
    -------
    x_log_probs_prox : torch.Tensor
        (num_states,). Each state's recorded action's log-prob, summed over the levels.
    x_value_prox : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values.
    x_value_final : torch.Tensor
        (num_envs, num_reward_streams). Each game's final value.
    """
    rollout_length, num_envs, num_reward_streams = rollout.x_rewards.shape
    num_states = rollout_length * num_envs
    
    x_batch_log_prob = []
    x_batch_value = []
    with torch.no_grad():
        for start in range(0, num_states, minibatch_size):
            stop = min(start + minibatch_size, num_states)

            # Slice `StateBatch` and actions
            state_batch_slice = slice_state_batch(
                rollout.state_batch, start, stop, state_batch_layout
            )
            x_actions_slice = rollout.x_actions[start:stop]

            # Evaluate actions on the slice's states
            x_log_prob, _, _, x_value = model.evaluate_actions(
                state_batch_slice, x_actions_slice
            )

            # Append log-probabilities and values
            x_batch_log_prob.append(x_log_prob)
            x_batch_value.append(x_value)

        # Bootstrap value i.e. value of the rollout's final state for each environment
        x_value_final = model(rollout.state_batch_final)

    x_value_prox = torch.cat(x_batch_value).reshape(
        rollout_length, num_envs, num_reward_streams
    )
    return torch.cat(x_batch_log_prob), x_value_prox, x_value_final


def _compute_gae(
    x_rewards: torch.Tensor,
    x_values: torch.Tensor,
    x_dones: torch.Tensor,
    x_value_final: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over `num_envs` parallel games, one recursion per reward stream.

    A done zeroes the next value and the carried advantage, so nothing leaks across an episode
    boundary.

    Parameters
    ----------
    x_rewards : torch.Tensor
        (rollout_length, num_envs, num_reward_streams) float32. Each step's rewards, one per
        reward stream.
    x_values : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values.
    x_dones : torch.Tensor
        (rollout_length, num_envs, 1) float32. Whether each step ended its game's episode: 1.0 if
        so, else 0.0.
    x_value_final : torch.Tensor
        (num_envs, num_reward_streams). Each game's value after the last step.
    gamma : float
        The discount.
    lam : float
        GAE's λ.

    Returns
    -------
    x_returns : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each step's advantages plus its values.
    x_advantages : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each step's advantages.
    """
    rollout_length = x_rewards.shape[0]

    # Initialize tensors
    x_advantages = torch.zeros_like(x_rewards)
    x_gae = torch.zeros_like(x_value_final)

    # Iterate in reversed order
    for t in reversed(range(rollout_length)):
        x_value = x_values[t]
        x_value_next = x_value_final if t == rollout_length - 1 else x_values[t + 1]

        # Game is still going flag
        x_ongoing = 1.0 - x_dones[t]

        # Calculate one-step TD error & GAE
        x_delta = x_rewards[t] + gamma * x_value_next * x_ongoing - x_value
        x_gae = x_delta + gamma * lam * x_ongoing * x_gae

        # Store GAE
        x_advantages[t] = x_gae

    return x_advantages + x_values, x_advantages


def _plan_minibatches(
    x_counts_cpu: torch.Tensor,
    minibatch_size: int,
    num_epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> list[MinibatchPlan]:
    """Plan every epoch before training, transferring all shuffled state IDs together.

    Parameters
    ----------
    x_counts_cpu : torch.Tensor
        (num_states, number_of_Jaggeds) on the CPU. Each state's row and entry counts, in Jagged
        order.
    minibatch_size : int
        The maximum number of states per minibatch.
    num_epochs : int
        The number of independently shuffled passes over the rollout.
    device : torch.device
        The device on which the minibatches will select their state rows.
    generator : torch.Generator
        The CPU generator used to shuffle each epoch's state order.

    Returns
    -------
    list[MinibatchPlan]
        The minibatches in the order they run, epoch after epoch, with device state indices and
        CPU total row and entry counts. Every epoch has the same number of them.
    """
    num_states = len(x_counts_cpu)

    # One shuffle of the states per epoch
    x_state_ids_cpu = torch.stack(
        [torch.randperm(num_states, generator=generator) for _ in range(num_epochs)]
    )
    x_state_ids = (x_state_ids_cpu.pin_memory() if device.type == "cuda" else x_state_ids_cpu).to(
        device, non_blocking=True
    )

    # Each shuffle cut into minibatches, their rows and entries totalled on the CPU
    minibatch_plans = []
    for num_epoch in range(num_epochs):
        for start in range(0, num_states, minibatch_size):
            stop = start + minibatch_size
            x_state_ids_cpu_minibatch = x_state_ids_cpu[num_epoch, start:stop]
            x_counts_cpu_minibatch = x_counts_cpu[x_state_ids_cpu_minibatch].sum(0, dtype=torch.long)
            minibatch_plans.append(MinibatchPlan(x_state_ids[num_epoch, start:stop], x_counts_cpu_minibatch.tolist()))

    return minibatch_plans


def _update_minibatch(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    state_batch_layout: StateBatchLayout,
    minibatch_plan: MinibatchPlan,
    targets: _Targets,
    ppo_config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Take one optimizer step on a minibatch of the rollout's states.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model, updated in place.
    optimizer : torch.optim.Optimizer
        The model's optimizer.
    rollout : Rollout
        The collected states and actions.
    state_batch_layout : StateBatchLayout
        The rollout's cached offsets.
    minibatch_plan : MinibatchPlan
        The minibatch's states.
    targets : _Targets
        What the update trains against, one row per state of the rollout.
    ppo_config : PPOConfig
        The PPO settings.
    device : torch.device
        The Learner's device.

    Returns
    -------
    dict[str, float]
        The minibatch's metrics, keyed by their TensorBoard scalar names; its approx KL is the
        ranks' mean.
    """
    x_state_ids = minibatch_plan.x_state_ids

    # Forward pass, under the weights so far
    state_batch_mini = take_minibatch_states(rollout.state_batch, minibatch_plan, state_batch_layout)
    x_log_probs, x_entropies, x_kl_uniform, x_values = model.evaluate_actions(
        state_batch_mini, rollout.x_actions[x_state_ids],
    )

    # Policy Loss
    x_log_ratios = x_log_probs - targets.x_log_probs_prox[x_state_ids]
    x_ratios = x_log_ratios.exp()
    x_loss_policy = _compute_loss_policy(
        x_ratios,
        targets.x_advantages[x_state_ids],
        targets.x_weights_capped[x_state_ids],
        ppo_config.clip_eps,
    )

    # Value loss: summed over streams, mean over the batch
    x_returns = targets.x_returns[x_state_ids]
    x_loss_value = 0.5 * F.mse_loss(x_values, x_returns, reduction="none").sum(-1).mean()
    x_loss_entropy = -torch.mean(x_entropies.sum(dim=-1))

    # KL-Uniform barrier, summed per head like the entropy term; each head's KL falls
    x_loss_kl_uniform = torch.mean(x_kl_uniform.sum(dim=-1))
    x_loss = (
        x_loss_policy
        + ppo_config.coef_value * x_loss_value
        + ppo_config.coef_entropy * x_loss_entropy
        + ppo_config.coef_kl_uniform * x_loss_kl_uniform
    )

    # Optimizer step, on the gradient averaged over the ranks
    optimizer.zero_grad()
    x_loss.backward()
    average_gradients(model)
    x_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), ppo_config.max_grad_norm)
    optimizer.step()

    # Compute metrics
    metrics = _minibatch_metrics(
        x_loss_policy,
        x_loss_value,
        x_loss_entropy,
        x_loss_kl_uniform,
        x_entropies,
        x_kl_uniform,
        x_ratios,
        x_log_ratios,
        x_grad_norm,
        ppo_config.clip_eps,
    )

    # Every rank stops on the same KL, or the others would wait in the next all-reduce
    approx_kl = mean_over_ranks(metrics[MetricKey.UPDATE_APPROX_KL], device)
    metrics[MetricKey.UPDATE_APPROX_KL] = approx_kl
    return metrics


def _compute_loss_policy(
    x_ratios: torch.Tensor,
    x_advantages: torch.Tensor,
    x_weights_capped: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Compute the policy loss of decoupled PPO (Hilton et al. 2021).

    It clips against π_prox, and weights each sample by π_prox / π_behavior, capped, for the
    weights that played.

    Parameters
    ----------
    x_ratios : torch.Tensor
        (batch_size,). Each recorded action's probability ratio, the current weights' over
        π_prox's.
    x_advantages : torch.Tensor
        (batch_size,). Each state's normalized advantage.
    x_weights_capped : torch.Tensor
        (batch_size,). Each state's capped off-policy weight.
    clip_eps : float
        The clip range.

    Returns
    -------
    torch.Tensor
        (). The policy loss.
    """
    x_surrogate = x_ratios * x_advantages
    x_surrogate_clipped = torch.clamp(x_ratios, 1 - clip_eps, 1 + clip_eps) * x_advantages
    return -torch.mean(x_weights_capped * torch.min(x_surrogate, x_surrogate_clipped))


def _minibatch_metrics(
    x_loss_policy: torch.Tensor,
    x_loss_value: torch.Tensor,
    x_loss_entropy: torch.Tensor,
    x_loss_kl_uniform: torch.Tensor,
    x_entropies: torch.Tensor,
    x_kl_uniform: torch.Tensor,
    x_ratios: torch.Tensor,
    x_log_ratios: torch.Tensor,
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
    x_loss_kl_uniform : torch.Tensor
        (). The uniform-KL loss.
    x_entropies : torch.Tensor
        (batch_size, 3). The entropy of each of the minibatch's states at each level.
    x_kl_uniform : torch.Tensor
        (batch_size, 3). Each state's KL(uniform‖π) at each level.
    x_ratios : torch.Tensor
        (batch_size,). Each recorded action's probability ratio, the current weights' over
        π_prox's.
    x_log_ratios : torch.Tensor
        (batch_size,). The ratios' logs.
    x_grad_norm : torch.Tensor
        (). The gradients' total norm before clipping.
    clip_eps : float
        The clip range.

    Returns
    -------
    dict[str, float]
        Each scalar under its TensorBoard name.
    """
    x_entropies_mean = x_entropies.mean(dim=0)
    x_kl_uniform_mean = x_kl_uniform.mean(dim=0)
    return _read_metrics(
        {
            # Loss
            MetricKey.LOSS_POLICY: x_loss_policy,
            MetricKey.LOSS_VALUE: x_loss_value,
            MetricKey.LOSS_ENTROPY: x_loss_entropy,
            MetricKey.LOSS_KL_UNIFORM: x_loss_kl_uniform,

            # Entropy
            MetricKey.ENTROPY_L1: x_entropies_mean[0],
            MetricKey.ENTROPY_L2: x_entropies_mean[1],
            MetricKey.ENTROPY_L3: x_entropies_mean[2],

            # KL-Uniform
            MetricKey.KL_UNIFORM_L1: x_kl_uniform_mean[0],
            MetricKey.KL_UNIFORM_L2: x_kl_uniform_mean[1],
            MetricKey.KL_UNIFORM_L3: x_kl_uniform_mean[2],

            # KL-Update
            MetricKey.UPDATE_APPROX_KL: ((x_ratios - 1) - x_log_ratios).mean(),

            # Other
            MetricKey.UPDATE_CLIP_FRACTION: ((x_ratios - 1).abs() > clip_eps).float().mean(),
            MetricKey.GRAD_NORM_PRECLIP: x_grad_norm,
        }
    )


def _left_trust_region(
    records: list[dict[str, float]], num_minibatches: int, target_kl: float
) -> bool:
    """Tell whether the last minibatch took the policy out of the trust region.

    Parameters
    ----------
    records : list[dict[str, float]]
        The metrics of each minibatch run so far, in order.
    num_minibatches : int
        The number of minibatches per epoch.
    target_kl : float
        The target KL.

    Returns
    -------
    bool
        True if the last minibatch's KL is above 3x the target, or if it closes an epoch whose
        mean KL is above 1.5x the target.
    """
    # Minibatch-level trust region: 3.0x the target
    if records[-1][MetricKey.UPDATE_APPROX_KL] > 3.0 * target_kl:
        return True

    # Batch-level trust region: 1.5x the target
    if len(records) % num_minibatches != 0:
        return False
    
    approx_kls = [metrics[MetricKey.UPDATE_APPROX_KL] for metrics in records[-num_minibatches:]]
    return sum(approx_kls) / num_minibatches > 1.5 * target_kl


def _average_minibatch_metrics(
    records: list[dict[str, float]], num_minibatches: int
) -> dict[str, float]:
    """Average the minibatches' metrics and count the epochs run.

    Parameters
    ----------
    records : list[dict[str, float]]
        The metrics of each minibatch run, in order.
    num_minibatches : int
        The number of minibatches per epoch.

    Returns
    -------
    dict[str, float]
        Each per-minibatch metric averaged over the minibatches run, and `update/epochs_run`.
    """
    metrics = {key: sum(metrics[key] for metrics in records) / len(records) for key in records[0]}
    metrics[MetricKey.UPDATE_EPOCHS_RUN] = len(records) / num_minibatches
    return metrics


def _anchor_metrics(
    x_returns: torch.Tensor,
    x_value_prox: torch.Tensor,
    x_log_weights: torch.Tensor,
    x_weights: torch.Tensor,
    is_clip: float,
) -> dict[str, float]:
    """Return the TensorBoard scalars computed once per update, over the whole rollout.

    Parameters
    ----------
    x_returns : torch.Tensor
        (num_states, num_reward_streams). Each state's returns.
    x_value_prox : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values under π_prox's
        weights.
    x_log_weights : torch.Tensor
        (num_states,). The log of each state's off-policy weight, π_prox / π_behavior, uncapped.
    x_weights : torch.Tensor
        (num_states,). Each state's off-policy weight, uncapped.
    is_clip : float
        The off-policy weights' cap.

    Returns
    -------
    dict[str, float]
        Each scalar under its TensorBoard name.
    """
    # Per-stream explained variance of the values the Learner recomputed before the update
    x_value_prox = x_value_prox.reshape(x_returns.shape)
    x_explained_var = 1.0 - (x_returns - x_value_prox).var(dim=0) / (x_returns.var(dim=0) + 1e-8)

    return _read_metrics(
        {
            **{f"{MetricKey.VALUE_EV}_{reward_stream}": x_explained_var[idx] for idx, reward_stream in enumerate(RewardStream)},
            MetricKey.OFFPOLICY_IS_WEIGHT_MEAN: x_weights.clamp(max=is_clip).mean(),
            MetricKey.OFFPOLICY_IS_WEIGHT_TRUNCATED: (x_weights > is_clip).float().mean(),
            MetricKey.OFFPOLICY_KL_BEHAVIOR_PROX: (x_weights - 1 - x_log_weights).mean(),
        }
    )


def _read_metrics(x_metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Read named 0-d tensors as floats in one transfer.

    Parameters
    ----------
    x_metrics : dict[str, torch.Tensor]
        (). Each scalar, on any one device, under its TensorBoard name.

    Returns
    -------
    dict[str, float]
        The same scalars as floats.
    """
    # One read for all of them: each .item() would make the CPU wait for the GPU
    values = torch.stack([x_metric.detach() for x_metric in x_metrics.values()]).tolist()
    return dict(zip(x_metrics, values))
