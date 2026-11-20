from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.algorithms.actor_critic.distributed import GradientBuckets
from src.rl.algorithms.actor_critic.distributed import RankMean
from src.rl.algorithms.actor_critic.distributed import average_gradient_buckets
from src.rl.algorithms.actor_critic.distributed import mean_over_ranks
from src.rl.algorithms.actor_critic.distributed import read_rank_mean
from src.rl.algorithms.actor_critic.metrics import MetricKey
from src.rl.batches import cat_states
from src.rl.batches import gather_states
from src.rl.batches import slice_states
from src.rl.maps import get_unique_maps
from src.rl.models import ActorCritic
from src.rl.models.entity_projector import dedup_tokens
from src.rl.models.entity_projector import plan_tokens
from src.rl.models.map_encoder import dedup_maps
from src.rl.models.map_encoder import plan_maps
from src.rl.reward import RewardStream
from src.rl.types import Live
from src.rl.types import MapDedup
from src.rl.types import MapPlan
from src.rl.types import Rollout
from src.rl.types import States
from src.rl.types import TokenDedup
from src.rl.types import TokenPlan


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

    x_log_prob_prox: torch.Tensor  # (batch_size,) the recorded actions' log-probs under π_prox
    x_return: torch.Tensor         # (batch_size, num_reward_streams)
    x_advantage: torch.Tensor      # (batch_size,) summed over the reward streams, normalized
    x_weight_capped: torch.Tensor  # (batch_size,) π_prox / π_behavior, capped


def update_ppo(
    model: ActorCritic,
    rollout: Rollout,
    optimizer: torch.optim.Optimizer,
    gradient_buckets: GradientBuckets,
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
    gradient_buckets : GradientBuckets
        The model's gradient buckets, which average each step's gradients over the ranks.
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
        variances, the `offpolicy/*` scalars and, with a target KL, `update/approx_kl`, the
        update's total KL, each computed once per update.
    """
    num_reward_streams = rollout.x_reward.shape[-1]

    map_plan, token_plan = _plan_dedup(rollout)

    # Calculate the rollout's log-probabilities and values under the current weights
    x_log_prob_prox, x_value_prox, x_value_final = _evaluate_rollout(
        model, rollout, ppo_config.minibatch_size, map_plan, token_plan, device
    )

    # Compute GAE returns and advantages
    x_return, x_advantage = _compute_gae(
        rollout.x_reward,
        x_value_prox,
        rollout.x_done.unsqueeze(-1),
        rollout.x_num_discount.unsqueeze(-1),
        x_value_final,
        gamma,
        ppo_config.lam,
    )

    # Sum per-stream advantages (advantage is linear) and normalize them over the batch
    x_advantage = x_advantage.sum(-1)
    x_advantage = (x_advantage - x_advantage.mean()) / (x_advantage.std() + 1e-8)

    # One row per state from here on
    x_log_weight = x_log_prob_prox - rollout.x_log_prob_behavior
    x_weight = x_log_weight.exp()
    targets = _Targets(
        x_log_prob_prox=x_log_prob_prox,
        x_return=x_return.reshape(-1, num_reward_streams),
        x_advantage=x_advantage.reshape(-1),
        x_weight_capped=x_weight.clamp(max=ppo_config.is_clip),
    )

    # All shuffles are drawn and transferred before the first minibatch launches
    idxs_minibatch = _plan_minibatches(
        len(rollout.states.x_context),
        ppo_config.minibatch_size,
        ppo_config.num_epochs,
        device,
        generator,
    )
    num_minibatches = len(idxs_minibatch) // ppo_config.num_epochs

    batch_x_metrics = []
    batch_approx_kl = []
    for indices in idxs_minibatch:
        # Select this minibatch's states, actions and targets
        states_mini = gather_states(rollout.states, indices)

        # Select played actions and targets
        x_idx_state_cpu, x_idx_state = indices
        x_action_mini = rollout.x_action[x_idx_state]
        targets_mini = _Targets(
            x_log_prob_prox=targets.x_log_prob_prox[x_idx_state],
            x_return=targets.x_return[x_idx_state],
            x_advantage=targets.x_advantage[x_idx_state],
            x_weight_capped=targets.x_weight_capped[x_idx_state],
        )

        # The minibatch's Maps and tokens, deduped from the plans
        np_idx_state = x_idx_state_cpu.numpy()
        map_dedup_mini = dedup_maps(map_plan, np_idx_state, device)
        token_dedup_mini = dedup_tokens(token_plan, np_idx_state, device)

        # Update policy and value networks
        x_metrics, rank_mean_approx_kl = _update_minibatch(
            model,
            optimizer,
            gradient_buckets,
            states_mini,
            x_action_mini,
            targets_mini,
            ppo_config,
            map_dedup_mini,
            token_dedup_mini,
        )
        batch_x_metrics.append(x_metrics)
        batch_approx_kl.append(read_rank_mean(rank_mean_approx_kl))

        # Trust-region early stop check
        if ppo_config.target_kl is not None and _left_trust_region(
            batch_approx_kl, num_minibatches, ppo_config.target_kl
        ):
            break

    # With a target KL, the update's approx KL: its total, from π_prox to the updated weights
    approx_kl_update: float | None = None
    if ppo_config.target_kl is not None:
        x_log_prob_updated, _, _ = _evaluate_rollout(
            model, rollout, ppo_config.minibatch_size, map_plan, token_plan, device
        )
        approx_kl_update = read_rank_mean(
            mean_over_ranks(_approx_kl(x_log_prob_updated - x_log_prob_prox))
        )

    # Each per-minibatch metric averaged over the minibatches run, then the once-per-update ones
    metrics_update = _average_minibatch_metrics(
        _read_metrics(*batch_x_metrics), num_minibatches
    ) | _anchor_metrics(targets.x_return, x_value_prox, x_log_weight, x_weight, ppo_config.is_clip)
    if approx_kl_update is not None:
        metrics_update[MetricKey.UPDATE_APPROX_KL] = approx_kl_update

    return metrics_update


def adapt_lr(lr: float, approx_kl: float, target_kl: float) -> float:
    """Return the next adaptive-KL learning rate.

    Parameters
    ----------
    lr : float
        The current learning rate.
    approx_kl : float
        The last update's total approx KL.
    target_kl : float
        The target KL.

    Returns
    -------
    float
        `lr` divided by 1.5 when `approx_kl` is above twice `target_kl`, down to 1e-5;
        multiplied by 1.5 when below half of it, up to 1e-3; unchanged otherwise.
    """
    if approx_kl > 2.0 * target_kl:
        return max(lr / 1.5, 1e-5)
    
    if approx_kl < 0.5 * target_kl:
        return min(lr * 1.5, 1e-3)
    
    return lr


def _plan_dedup(rollout: Rollout) -> tuple[MapPlan, TokenPlan]:
    """Build both host plans from the same Map identities over all the rollout's states."""
    states = cat_states([rollout.states, rollout.states_final])
    maps = get_unique_maps(states)

    return plan_maps(maps), plan_tokens(states, maps.x_room_key)


def _evaluate_rollout(
    model: ActorCritic,
    rollout: Rollout,
    minibatch_size: int,
    map_plan: MapPlan,
    token_plan: TokenPlan,
    device: torch.device,
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
    map_plan : MapPlan
        The rollout's unique Maps, which each forward's Maps are deduped from.
    token_plan : TokenPlan
        The rollout's unique tokens, which each forward's tokens are deduped from.
    device : torch.device
        The Learner's device.

    Returns
    -------
    x_log_prob : torch.Tensor
        (batch_size,). Each state's recorded action's log-prob, summed over the levels.
    x_value : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values.
    x_value_final : torch.Tensor
        (num_envs, num_reward_streams). Each game's final value.
    """
    rollout_length, num_envs, num_reward_streams = rollout.x_reward.shape
    batch_size = rollout_length * num_envs
    
    batch_x_log_prob = []
    batch_x_value = []
    with torch.no_grad():
        for idx_start in range(0, batch_size, minibatch_size):
            idx_stop = min(idx_start + minibatch_size, batch_size)

            # Slice `States` and actions
            states_slice = slice_states(rollout.states, idx_start, idx_stop)
            x_action_slice = rollout.x_action[idx_start:idx_stop]

            # The slice's Maps and tokens, deduped from the plans
            np_idx_state = np.arange(idx_start, idx_stop)
            map_dedup_slice = dedup_maps(map_plan, np_idx_state, device)
            token_dedup_slice = dedup_tokens(token_plan, np_idx_state, device)

            # Evaluate actions on the slice's states
            policy_output, x_value_slice = model.action_evaluate(
                states_slice, x_action_slice, map_dedup_slice, token_dedup_slice
            )

            # Append log-probabilities and values
            batch_x_log_prob.append(policy_output.x_log_prob.sum(-1))
            batch_x_value.append(x_value_slice)

        # Bootstrap value i.e. value of the rollout's final state for each environment
        states_final = rollout.states_final
        np_idx_state_final = np.arange(batch_size, batch_size + num_envs)
        map_dedup_final = dedup_maps(map_plan, np_idx_state_final, device)
        token_dedup_final = dedup_tokens(token_plan, np_idx_state_final, device)
        x_value_final = model.get_values(states_final, map_dedup_final, token_dedup_final)

    x_value = torch.cat(batch_x_value).reshape(rollout_length, num_envs, num_reward_streams)

    return torch.cat(batch_x_log_prob), x_value, x_value_final


def _compute_gae(
    x_reward: torch.Tensor,
    x_value: torch.Tensor,
    x_done: torch.Tensor,
    x_num_discount: torch.Tensor,
    x_value_final: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over `num_envs` parallel games, one recursion per reward stream.

    A done zeroes the next value and the carried advantage, so nothing leaks across an episode
    boundary.

    Parameters
    ----------
    x_reward : torch.Tensor
        (rollout_length, num_envs, num_reward_streams) float32. Each step's rewards, one per
        reward stream.
    x_value : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values.
    x_done : torch.Tensor
        (rollout_length, num_envs, 1) float32. Whether each step ended its game's episode: 1.0 if
        so, else 0.0.
    x_num_discount : torch.Tensor
        (rollout_length, num_envs, 1) long. Each step's k: it discounts by γ^k and decays the
        trace by (γλ)^k.
    x_value_final : torch.Tensor
        (num_envs, num_reward_streams). Each game's value after the last step.
    gamma : float
        The discount, γ.
    lam : float
        GAE's λ.

    Returns
    -------
    x_return : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each step's advantages plus its values.
    x_advantage : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each step's advantages.
    """
    rollout_length = x_reward.shape[0]

    # Initialize tensors
    x_advantage = torch.zeros_like(x_reward)
    x_gae = torch.zeros_like(x_value_final)

    # Iterate in reversed order
    for idx_step in reversed(range(rollout_length)):
        x_value_step = x_value[idx_step]
        x_value_next = x_value_final if idx_step == rollout_length - 1 else x_value[idx_step + 1]

        # Game is still going flag
        x_ongoing = 1.0 - x_done[idx_step]

        # Calculate one-step TD error & GAE
        x_discount = gamma ** x_num_discount[idx_step]
        x_delta = x_reward[idx_step] + x_discount * x_value_next * x_ongoing - x_value_step
        x_gae = x_delta + x_discount * lam ** x_num_discount[idx_step] * x_ongoing * x_gae

        # Store GAE
        x_advantage[idx_step] = x_gae

    return x_advantage + x_value, x_advantage


def _plan_minibatches(
    batch_size: int,
    minibatch_size: int,
    num_epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Plan every epoch before training, transferring all shuffled state indexes together.

    Parameters
    ----------
    batch_size : int
        The number of states in the rollout.
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
    list[tuple[torch.Tensor, torch.Tensor]]
        Each minibatch's state indexes on the CPU and on the device, in the order the minibatches
        run, epoch after epoch. Every epoch has the same number of them.
    """
    # One shuffle of the states per epoch
    x_idx_state_cpu = torch.stack(
        [torch.randperm(batch_size, generator=generator) for _ in range(num_epochs)]
    )
    x_idx_state = (x_idx_state_cpu.pin_memory() if device.type == "cuda" else x_idx_state_cpu).to(
        device, non_blocking=True
    )

    # Each shuffle cut into minibatches
    idxs_minibatch = []
    for idx_epoch in range(num_epochs):
        for idx_start in range(0, batch_size, minibatch_size):
            idx_stop = idx_start + minibatch_size
            x_idx_state_cpu_mini = x_idx_state_cpu[idx_epoch, idx_start:idx_stop]
            idxs_minibatch.append(
                (
                    x_idx_state_cpu_mini,
                    x_idx_state[idx_epoch, idx_start:idx_stop],
                )
            )

    return idxs_minibatch


def _update_minibatch(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    gradient_buckets: GradientBuckets,
    states: States[Live],
    x_action: torch.Tensor,
    targets: _Targets,
    ppo_config: PPOConfig,
    map_dedup: MapDedup,
    token_dedup: TokenDedup | None,
) -> tuple[dict[str, torch.Tensor], RankMean]:
    """Take one optimizer step on the selected states, actions and targets.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model, updated in place.
    optimizer : torch.optim.Optimizer
        The model's optimizer.
    gradient_buckets : GradientBuckets
        The model's gradient buckets, which average each step's gradients over the ranks.
    states : States[Live]
        The minibatch's selected states, including their CPU totals.
    x_action : torch.Tensor
        (minibatch_size, 3) long. The recorded actions for these states.
    targets : _Targets
        What the update trains against, one row per selected state.
    ppo_config : PPOConfig
        The PPO settings.
    map_dedup : MapDedup
        The Maps the Map GNN runs for these states.
    token_dedup : TokenDedup | None
        The unique tokens the projector runs for these states, or None to run every token.

    Returns
    -------
    x_metrics : dict[str, torch.Tensor]
        The minibatch's metrics, keyed by their TensorBoard scalar names, unread on the device.
    rank_mean_approx_kl : RankMean
        The approx KL from π_prox to the weights before this step, averaged over the ranks: the
        early stops' input, on its way to the host.
    """
    # Forward pass, under the weights so far
    policy_output, x_value = model.action_evaluate(states, x_action, map_dedup, token_dedup)

    # Policy Loss
    x_log_ratio = policy_output.x_log_prob.sum(-1) - targets.x_log_prob_prox
    x_ratio = x_log_ratio.exp()
    x_loss_policy = _compute_loss_policy(
        x_ratio,
        targets.x_advantage,
        targets.x_weight_capped,
        ppo_config.clip_eps,
    )

    # Value loss: summed over streams, mean over the batch
    x_return = targets.x_return
    x_loss_value = 0.5 * F.mse_loss(x_value, x_return, reduction="none").sum(-1).mean()
    x_loss_entropy = -torch.mean(policy_output.x_entropy.sum(dim=-1))

    # KL-Uniform barrier, summed per level like the entropy term; each level's KL falls
    x_loss_kl_uniform = torch.mean(policy_output.x_kl_uniform.sum(dim=-1))
    x_loss = (
        x_loss_policy
        + ppo_config.coef_value * x_loss_value
        + ppo_config.coef_entropy * x_loss_entropy
        + ppo_config.coef_kl_uniform * x_loss_kl_uniform
    )

    # Every rank stops on the same KL, or the others would wait in the next all-reduce
    rank_mean_approx_kl = mean_over_ranks(_approx_kl(x_log_ratio))

    # Optimizer step, on the gradient averaged over the ranks
    optimizer.zero_grad()
    x_loss.backward()
    average_gradient_buckets(gradient_buckets, model)
    x_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), ppo_config.max_grad_norm)
    optimizer.step()

    # Compute metrics
    x_metrics = _minibatch_metrics(
        x_loss_policy,
        x_loss_value,
        x_loss_entropy,
        x_loss_kl_uniform,
        policy_output.x_entropy,
        policy_output.x_kl_uniform,
        x_ratio,
        x_grad_norm,
        ppo_config.clip_eps,
    )

    return x_metrics, rank_mean_approx_kl


def _compute_loss_policy(
    x_ratio: torch.Tensor,
    x_advantage: torch.Tensor,
    x_weight_capped: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Compute the policy loss of decoupled PPO (Hilton et al. 2021).

    It clips against π_prox, and weights each state by π_prox / π_behavior, capped, for the
    weights that played.

    Parameters
    ----------
    x_ratio : torch.Tensor
        (batch_size,). Each recorded action's probability ratio, the current weights' over
        π_prox's.
    x_advantage : torch.Tensor
        (batch_size,). Each state's normalized advantage.
    x_weight_capped : torch.Tensor
        (batch_size,). Each state's capped off-policy weight.
    clip_eps : float
        The clip range.

    Returns
    -------
    torch.Tensor
        (). The policy loss.
    """
    x_surrogate = x_ratio * x_advantage
    x_surrogate_clipped = torch.clamp(x_ratio, 1 - clip_eps, 1 + clip_eps) * x_advantage

    return -torch.mean(x_weight_capped * torch.min(x_surrogate, x_surrogate_clipped))


def _minibatch_metrics(
    x_loss_policy: torch.Tensor,
    x_loss_value: torch.Tensor,
    x_loss_entropy: torch.Tensor,
    x_loss_kl_uniform: torch.Tensor,
    x_entropy: torch.Tensor,
    x_kl_uniform: torch.Tensor,
    x_ratio: torch.Tensor,
    x_grad_norm: torch.Tensor,
    clip_eps: float,
) -> dict[str, torch.Tensor]:
    """Return the per-minibatch TensorBoard scalars (minibatch means), unread on their device.

    Clip fraction = share of moved-off ratios; grad norm is the pre-clip total (otherwise
    discarded by clip_grad_norm_).

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
    x_entropy : torch.Tensor
        (batch_size, 2). The entropy of each of the minibatch's states at each level.
    x_kl_uniform : torch.Tensor
        (batch_size, 2). Each state's KL-Uniform at each level.
    x_ratio : torch.Tensor
        (batch_size,). Each recorded action's probability ratio, the current weights' over
        π_prox's.
    x_grad_norm : torch.Tensor
        (). The gradients' total norm before clipping.
    clip_eps : float
        The clip range.

    Returns
    -------
    dict[str, torch.Tensor]
        (). Each scalar under its TensorBoard name.
    """
    x_entropy_mean = x_entropy.mean(dim=0)
    x_kl_uniform_mean = x_kl_uniform.mean(dim=0)

    return {
        # Loss
        MetricKey.LOSS_POLICY: x_loss_policy,
        MetricKey.LOSS_VALUE: x_loss_value,
        MetricKey.LOSS_ENTROPY: x_loss_entropy,
        MetricKey.LOSS_KL_UNIFORM: x_loss_kl_uniform,

        # Entropy
        MetricKey.ENTROPY_L1: x_entropy_mean[0],
        MetricKey.ENTROPY_L2: x_entropy_mean[1],

        # KL-Uniform
        MetricKey.KL_UNIFORM_L1: x_kl_uniform_mean[0],
        MetricKey.KL_UNIFORM_L2: x_kl_uniform_mean[1],

        # Other
        MetricKey.UPDATE_CLIP_FRACTION: ((x_ratio - 1).abs() > clip_eps).float().mean(),
        MetricKey.GRAD_NORM_PRECLIP: x_grad_norm,
    }


def _left_trust_region(
    batch_approx_kl: list[float], num_minibatches: int, target_kl: float
) -> bool:
    """Tell whether the last minibatch took the policy out of the trust region.

    Parameters
    ----------
    batch_approx_kl : list[float]
        The approx KL of each minibatch run so far, in order, each from before its step.
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
    if batch_approx_kl[-1] > 3.0 * target_kl:
        return True

    # Batch-level trust region: 1.5x the target
    if len(batch_approx_kl) % num_minibatches != 0:
        return False
    
    return sum(batch_approx_kl[-num_minibatches:]) / num_minibatches > 1.5 * target_kl


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
    x_return: torch.Tensor,
    x_value_prox: torch.Tensor,
    x_log_weight: torch.Tensor,
    x_weight: torch.Tensor,
    is_clip: float,
) -> dict[str, float]:
    """Return the TensorBoard scalars computed once per update, over the whole rollout.

    Parameters
    ----------
    x_return : torch.Tensor
        (batch_size, num_reward_streams). Each state's returns.
    x_value_prox : torch.Tensor
        (rollout_length, num_envs, num_reward_streams). Each state's values under π_prox's
        weights.
    x_log_weight : torch.Tensor
        (batch_size,). The log of each state's off-policy weight, π_prox / π_behavior, uncapped.
    x_weight : torch.Tensor
        (batch_size,). Each state's off-policy weight, uncapped.
    is_clip : float
        The off-policy weights' cap.

    Returns
    -------
    dict[str, float]
        Each scalar under its TensorBoard name.
    """
    # Per-stream explained variance of the values the Learner recomputed before the update
    x_value_prox = x_value_prox.reshape(x_return.shape)
    x_explained_var = 1.0 - (x_return - x_value_prox).var(dim=0) / (x_return.var(dim=0) + 1e-8)

    return _read_metrics(
        {
            **{f"{MetricKey.VALUE_EV}_{reward_stream}": x_explained_var[idx_reward_stream] for idx_reward_stream, reward_stream in enumerate(RewardStream)},
            MetricKey.OFFPOLICY_IS_WEIGHT_MEAN: x_weight.clamp(max=is_clip).mean(),
            MetricKey.OFFPOLICY_IS_WEIGHT_TRUNCATED: (x_weight > is_clip).float().mean(),
            MetricKey.OFFPOLICY_KL_BEHAVIOR_PROX: _approx_kl(x_log_weight),
        }
    )[0]


def _approx_kl(x_log_ratio: torch.Tensor) -> torch.Tensor:
    """Return Schulman's approx-KL estimate: the mean of (r - 1) - log r over the log-ratios.

    Parameters
    ----------
    x_log_ratio : torch.Tensor
        (batch_size,). Each recorded action's log-ratio: the newer weights' log-prob minus the
        older weights'.

    Returns
    -------
    torch.Tensor
        (). The estimate of KL(older ‖ newer).
    """
    return ((x_log_ratio.exp() - 1) - x_log_ratio).mean()


def _read_metrics(*batch_x_metrics: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    """Read records of named 0-d tensors as floats in one transfer.

    Parameters
    ----------
    *batch_x_metrics : dict[str, torch.Tensor]
        (). Each record's scalars, all on one device, each under its TensorBoard name; every
        record has as many.

    Returns
    -------
    list[dict[str, float]]
        The same records, their scalars as floats.
    """
    # One read for all of them: each .item() would make the CPU wait for the GPU
    batch_values = torch.stack(
        [
            torch.stack([x_metric.detach() for x_metric in x_metrics.values()])
            for x_metrics in batch_x_metrics
        ]
    ).tolist()

    return [
        dict(zip(x_metrics, values)) for x_metrics, values in zip(batch_x_metrics, batch_values)
    ]
