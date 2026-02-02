"""
Master process for PPO training with parallel workers.

Coordinates multiple worker processes, collects trajectories, and updates the model.
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pipe
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

from src.game.view.state import ViewGameState
from src.rl.action_space import ActionChoice
from src.rl.action_space import CHOICE_TO_HEAD
from src.rl.action_space import HeadType
from src.rl.action_space.masks import get_masks_batch
from src.rl.algorithms.actor_critic.worker import Command
from src.rl.algorithms.actor_critic.worker import WorkerData
from src.rl.algorithms.actor_critic.worker import worker
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models import _slice_core_output
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Transition:
    """Single transition in a trajectory."""

    view_game_state: ViewGameState  # Store view state, encode lazily
    action_choice: ActionChoice
    action_choice_log_prob: torch.Tensor
    secondary_index: int  # -1 if terminal
    secondary_log_prob: torch.Tensor
    value: torch.Tensor
    reward: float


@dataclass
class Trajectory:
    """Sequence of transitions from one episode."""

    transitions: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class TrajectoryBatch:
    """Batched trajectory data for training."""

    view_game_states: list[ViewGameState]
    action_choices: torch.Tensor  # (N,) ActionChoice indices
    action_choice_log_probs: torch.Tensor  # (N,)
    secondary_indices: torch.Tensor  # (N,) -1 if terminal
    secondary_log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, 1)
    returns: torch.Tensor  # (N, 1)
    advantages: torch.Tensor  # (N, 1)

    def __len__(self) -> int:
        return len(self.view_game_states)


# =============================================================================
# Episode Collection
# =============================================================================


def _run_episodes(
    model: ActorCritic,
    conn_parents: list[Connection],
    device: torch.device,
) -> list[Trajectory]:
    """
    Run episodes in parallel across all workers.

    Returns a list of trajectories, one per worker.
    """
    num_envs = len(conn_parents)

    # Reset all environments
    for conn in conn_parents:
        conn.send((Command.RESET, None))

    # Get initial states
    worker_datas: list[WorkerData] = [conn.recv() for conn in conn_parents]

    # Initialize trajectories
    trajectories = [Trajectory() for _ in range(num_envs)]

    # Track which envs are still running
    running_envs = list(range(num_envs))

    model.eval()
    with torch.no_grad():
        while running_envs:
            # Collect non-terminal states
            active_envs = []
            view_game_states = []

            for env_idx in running_envs:
                data = worker_datas[env_idx]
                if not data.game_over:
                    active_envs.append(env_idx)
                    view_game_states.append(data.view_game_state)

            running_envs = active_envs
            if not running_envs:
                break

            # Encode all states at once
            x_game_state = encode_batch_view_game_state(view_game_states, device)

            # Get masks for all states
            primary_mask, secondary_masks = get_masks_batch(view_game_states, device)

            # Forward pass (all samples, no FSM grouping needed!)
            output = model(x_game_state, primary_mask, secondary_masks, sample=True)

            # Send actions to workers
            for i, env_idx in enumerate(active_envs):
                action = output.get_action(i)
                conn_parents[env_idx].send((Command.STEP, action))

            # Receive new states
            new_worker_datas = {}
            for env_idx in running_envs:
                new_worker_datas[env_idx] = conn_parents[env_idx].recv()

            # Store transitions
            for i, env_idx in enumerate(running_envs):
                reward = new_worker_datas[env_idx].reward

                transition = Transition(
                    view_game_state=view_game_states[i],
                    action_choice=ActionChoice(output.action_choices[i].item()),
                    action_choice_log_prob=output.action_choice_log_probs[i],
                    secondary_index=output.secondary_indices[i].item(),
                    secondary_log_prob=output.secondary_log_probs[i],
                    value=output.values[i],
                    reward=reward,
                )
                trajectories[env_idx].transitions.append(transition)

            # Update worker data
            for env_idx in running_envs:
                worker_datas[env_idx] = new_worker_datas[env_idx]

    model.train()
    return trajectories


# =============================================================================
# GAE and Batch Creation
# =============================================================================


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Compute returns and GAE advantages."""
    returns = deque()
    advantages = deque()

    values_with_terminal = values + [torch.tensor([[0.0]], device=device)]

    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t]
            + gamma * values_with_terminal[t + 1].item()
            - values_with_terminal[t].item()
        )
        gae = delta + gamma * lam * gae
        returns.appendleft(gae + values_with_terminal[t].item())
        advantages.appendleft(gae)

    return list(returns), list(advantages)


def _create_batch(
    trajectories: list[Trajectory],
    gamma: float,
    lam: float,
    device: torch.device,
) -> TrajectoryBatch:
    """Create a training batch from trajectories."""
    all_view_states = []
    all_action_choices = []
    all_action_choice_log_probs = []
    all_secondary_indices = []
    all_secondary_log_probs = []
    all_values = []
    all_returns = []
    all_advantages = []

    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue

        rewards = [t.reward for t in trajectory.transitions]
        values = [t.value for t in trajectory.transitions]
        returns, advantages = _compute_gae(rewards, values, gamma, lam, device)

        for i, trans in enumerate(trajectory.transitions):
            all_view_states.append(trans.view_game_state)
            all_action_choices.append(int(trans.action_choice))
            all_action_choice_log_probs.append(trans.action_choice_log_prob.item())
            all_secondary_indices.append(trans.secondary_index)
            all_secondary_log_probs.append(trans.secondary_log_prob.item())
            all_values.append(trans.value)
            all_returns.append(returns[i])
            all_advantages.append(advantages[i])

    batch = TrajectoryBatch(
        view_game_states=all_view_states,
        action_choices=torch.tensor(all_action_choices, dtype=torch.long, device=device),
        action_choice_log_probs=torch.tensor(
            all_action_choice_log_probs, dtype=torch.float32, device=device
        ),
        secondary_indices=torch.tensor(all_secondary_indices, dtype=torch.long, device=device),
        secondary_log_probs=torch.tensor(
            all_secondary_log_probs, dtype=torch.float32, device=device
        ),
        values=torch.cat(all_values, dim=0).detach(),
        returns=torch.tensor(all_returns, dtype=torch.float32, device=device).view(-1, 1),
        advantages=torch.tensor(all_advantages, dtype=torch.float32, device=device).view(-1, 1),
    )

    # Normalize advantages
    batch.advantages = (batch.advantages - torch.mean(batch.advantages)) / (
        torch.std(batch.advantages) + 1e-8
    )

    return batch


# =============================================================================
# PPO Update
# =============================================================================


def _minibatch_indices(total: int, minibatch_size: int) -> Iterator[list[int]]:
    """Generate shuffled minibatch indices."""
    indices = list(range(total))
    random.shuffle(indices)

    for i in range(0, total, minibatch_size):
        yield indices[i : i + minibatch_size]


def _recompute_log_probs_batch(
    model: ActorCritic,
    view_game_states: list[ViewGameState],
    action_choices: torch.Tensor,
    secondary_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute log probs and entropy for a batch with current policy.

    Efficient batched implementation:
    1. Encode all states at once
    2. Run core encoder once
    3. Run primary head once
    4. Run secondary heads grouped by type

    Returns (log_probs, entropies, values) - all shape (N,) or (N, 1)
    """
    B = len(view_game_states)

    # Encode all states
    x_game_state = encode_batch_view_game_state(view_game_states, device)

    # Get masks
    primary_mask, secondary_masks = get_masks_batch(view_game_states, device)

    # Core encoder (all samples)
    core_out = model.core(x_game_state)

    # Value head (all samples)
    values = model.head_value(core_out.x_global)

    # Primary head (all samples) - get distributions
    primary_out = model.head_primary(core_out.x_global, primary_mask, sample=False)
    primary_dist = torch.distributions.Categorical(logits=primary_out.logits)
    primary_log_probs = primary_dist.log_prob(action_choices)
    primary_entropies = primary_dist.entropy()

    # Secondary heads - group by head type
    secondary_log_probs = torch.zeros(B, device=device)
    secondary_entropies = torch.zeros(B, device=device)

    head_to_samples: dict[HeadType, list[int]] = {ht: [] for ht in HeadType}
    for i in range(B):
        choice = ActionChoice(action_choices[i].item())
        head_type = CHOICE_TO_HEAD[choice]
        if head_type is not None and secondary_indices[i].item() >= 0:
            head_to_samples[head_type].append(i)

    for head_type, sample_idxs in head_to_samples.items():
        if not sample_idxs:
            continue

        idx = torch.tensor(sample_idxs, device=device)
        subset_core = _slice_core_output(core_out, idx)
        subset_mask = secondary_masks[head_type][idx]
        subset_secondary_idx = secondary_indices[idx]

        # Run head
        _, log_probs, out = _run_secondary_head_for_training(
            model, head_type, subset_core, subset_mask, subset_secondary_idx, device
        )

        secondary_log_probs[idx] = log_probs
        secondary_entropies[idx] = out.entropy()

    total_log_probs = primary_log_probs + secondary_log_probs
    total_entropies = primary_entropies + secondary_entropies

    return total_log_probs, total_entropies, values


def _run_secondary_head_for_training(
    model: ActorCritic,
    head_type: HeadType,
    core_out,
    mask: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
):
    """Run a secondary head and compute log probs for given indices."""
    if head_type == HeadType.MAP_SELECT:
        out = model.head_map_select(core_out.x_map, core_out.x_global, mask, sample=False)
    else:
        head = model._heads[head_type]
        entities = model._get_entities(head_type, core_out)
        out = head(entities, core_out.x_global, mask, sample=False)

    dist = torch.distributions.Categorical(logits=out.logits)
    log_probs = dist.log_prob(indices)

    return indices, log_probs, dist


def _update_ppo(
    model: ActorCritic,
    batch: TrajectoryBatch,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """PPO update step."""
    total_loss_policy = 0.0
    total_loss_value = 0.0
    total_loss_entropy = 0.0
    num_updates = 0

    for _ in range(num_epochs):
        for mb_idxs in _minibatch_indices(len(batch), minibatch_size):
            # Gather minibatch data
            mb_states = [batch.view_game_states[i] for i in mb_idxs]
            mb_action_choices = batch.action_choices[mb_idxs]
            mb_secondary_indices = batch.secondary_indices[mb_idxs]

            # Recompute with current policy
            log_probs_new, entropies, values_new = _recompute_log_probs_batch(
                model, mb_states, mb_action_choices, mb_secondary_indices, device
            )

            # Old log probs
            log_probs_old = (
                batch.action_choice_log_probs[mb_idxs] + batch.secondary_log_probs[mb_idxs]
            )

            # Advantages and returns
            advantages = torch.squeeze(batch.advantages[mb_idxs])
            returns = batch.returns[mb_idxs]
            values_old = batch.values[mb_idxs]

            # Policy loss (PPO clipped)
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss_policy = -torch.mean(torch.min(surr1, surr2))

            # Value loss
            if clip_value_loss:
                values_clipped = values_old + torch.clamp(
                    values_new - values_old, -clip_eps, clip_eps
                )
                loss_value_unclipped = torch.pow(values_new - returns, 2)
                loss_value_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.mean(torch.max(loss_value_unclipped, loss_value_clipped))
            else:
                loss_value = F.mse_loss(values_new, returns)

            # Entropy loss
            loss_entropy = -torch.mean(entropies)

            # Total loss
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_policy += loss_policy.item()
            total_loss_value += loss_value.item()
            total_loss_entropy += loss_entropy.item()
            num_updates += 1

    return (
        total_loss_policy / num_updates,
        total_loss_value / num_updates,
        total_loss_entropy / num_updates,
    )


# =============================================================================
# Training Loop
# =============================================================================


def _get_entropy_schedule(
    num_episodes: int,
    elbow: int,
    max_coef: float,
    min_coef: float,
) -> list[float]:
    """Linear decay of entropy coefficient."""
    coefs = []
    slope = (min_coef - max_coef) / elbow

    for ep in range(num_episodes):
        if ep <= elbow:
            coefs.append(slope * ep + max_coef)
        else:
            coefs.append(min_coef)

    return coefs


def train(
    exp_name: str,
    num_episodes: int,
    log_every: int,
    save_every: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    gamma: float,
    lam: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
    num_envs: int,
    device: torch.device,
) -> None:
    """Main training loop."""
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)

    # Start worker processes
    conn_parents, conn_children = zip(*[Pipe() for _ in range(num_envs)])
    processes = [Process(target=worker, args=(conn,)) for conn in conn_children]
    for p in processes:
        p.start()

    try:
        for episode in range(num_episodes):
            coef_entropy = coefs_entropy[episode]

            # Collect trajectories
            trajectories = _run_episodes(model, list(conn_parents), device)

            # Create batch
            batch = _create_batch(trajectories, gamma, lam, device)

            if len(batch) == 0:
                print(f"Episode {episode}: Empty batch, skipping")
                continue

            # PPO update
            loss_policy, loss_value, loss_entropy = _update_ppo(
                model,
                batch,
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

            # Logging
            if episode % log_every == 0:
                print(f"Episode {episode}: policy={loss_policy:.4f}, value={loss_value:.4f}")
                writer.add_scalar("Loss/policy", loss_policy, episode)
                writer.add_scalar("Loss/value", loss_value, episode)
                writer.add_scalar("Loss/entropy", loss_entropy, episode)
                writer.add_scalar("Entropy/coef", coef_entropy, episode)

                total_reward = sum(
                    sum(t.reward for t in traj.transitions) for traj in trajectories
                )
                avg_length = sum(len(traj) for traj in trajectories) / max(len(trajectories), 1)
                writer.add_scalar("Trajectory/total_reward", total_reward, episode)
                writer.add_scalar("Trajectory/avg_length", avg_length, episode)

            # Save
            if episode % save_every == 0:
                torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    finally:
        for conn in conn_parents:
            conn.send((Command.CLOSE, None))
        for p in processes:
            p.join()

    writer.close()


if __name__ == "__main__":
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)

    # Model
    model = ActorCritic(**config["model"])

    # Optimizer
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])

    # Create experiment directory
    os.makedirs(f"experiments/{config['exp_name']}", exist_ok=True)
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    # Entropy schedule
    coefs_entropy = _get_entropy_schedule(
        int(config["num_episodes"]),
        int(config["coef_entropy_elbow"]),
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )

    # Train
    print(f"Starting training: {config['exp_name']}")
    train(
        config["exp_name"],
        int(config["num_episodes"]),
        config["log_every"],
        config["save_every"],
        model,
        optimizer,
        config["num_epochs"],
        config["minibatch_size"],
        config["clip_eps"],
        config["clip_value_loss"],
        config["gamma"],
        config["lam"],
        config["coef_value"],
        coefs_entropy,
        config["max_grad_norm"],
        config["num_envs"],
        torch.device("cpu"),
    )
