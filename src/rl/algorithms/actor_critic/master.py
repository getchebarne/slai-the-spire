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

from src.game.action import Action
from src.game.action import ActionType
from src.game.const import MAP_WIDTH
from src.game.core.fsm import FSM
from src.rl.action_space import FSM_ROUTING
from src.rl.action_space import HeadType
from src.rl.action_space import get_secondary_head_type
from src.rl.action_space.masks import get_valid_mask_batch
from src.rl.algorithms.actor_critic.worker import Command
from src.rl.algorithms.actor_critic.worker import WorkerData
from src.rl.algorithms.actor_critic.worker import worker
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import ActorOutput
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


@dataclass
class Transition:
    x_game_state: XGameState
    fsm: FSM
    action_type: ActionType
    action_type_log_prob: torch.Tensor | None
    secondary_index: int | None
    secondary_log_prob: torch.Tensor | None
    value: torch.Tensor
    reward: float


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class TrajectoryBatch:
    """Batched trajectory data for training."""

    # Encoded states (each field is a tensor)
    x_game_states: list[XGameState]
    fsms: list[FSM]

    # Actions
    action_types: list[ActionType]
    action_type_indices: torch.Tensor  # (N,) index within route.action_types
    action_type_log_probs: torch.Tensor  # (N,) log prob (0 if forced)
    secondary_indices: torch.Tensor  # (N,) secondary head index (-1 if none)
    secondary_log_probs: torch.Tensor  # (N,) log prob (0 if none)

    # Values and returns
    values: torch.Tensor  # (N, 1)
    returns: torch.Tensor  # (N, 1)
    advantages: torch.Tensor  # (N, 1)

    def __len__(self) -> int:
        return len(self.x_game_states)


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
            fsms = []

            for env_idx in running_envs:
                data = worker_datas[env_idx]
                if not data.game_over:
                    active_envs.append(env_idx)
                    view_game_states.append(data.view_game_state)
                    fsms.append(data.fsm)

            # Remove terminal envs
            running_envs = active_envs
            if not running_envs:
                break

            # Process each FSM group separately (can't batch different FSM states)
            env_to_action: dict[int, Action] = {}
            env_to_output: dict[int, tuple[ActorOutput, torch.Tensor]] = {}

            # Group by FSM state
            fsm_groups: dict[FSM, list[int]] = {}
            for i, (env_idx, fsm) in enumerate(zip(active_envs, fsms)):
                if fsm not in fsm_groups:
                    fsm_groups[fsm] = []
                fsm_groups[fsm].append(i)

            # Process each FSM group (batched - model handles different action types)
            for fsm, batch_indices in fsm_groups.items():
                route = FSM_ROUTING[fsm]
                batch_states = [view_game_states[i] for i in batch_indices]
                batch_env_ids = [active_envs[i] for i in batch_indices]

                # Encode states
                x_game_state = encode_batch_view_game_state(batch_states, device)

                # Get masks for action type head
                masks = {}
                masks[HeadType.ACTION_TYPE] = get_valid_mask_batch(
                    HeadType.ACTION_TYPE, batch_states, device
                )

                # Get masks for all potential secondary heads
                for action_type in route.action_types:
                    secondary_head = get_secondary_head_type(fsm, action_type)
                    if secondary_head is not None and secondary_head not in masks:
                        masks[secondary_head] = get_valid_mask_batch(
                            secondary_head, batch_states, device
                        )

                # Forward pass (batched - handles different action types per sample)
                output = model.forward_batch(x_game_state, fsm, masks, sample=True)

                # Extract per-sample results
                for i, env_idx in enumerate(batch_env_ids):
                    env_to_action[env_idx] = output.actor.to_action(i)
                    env_to_output[env_idx] = (
                        output.actor.to_actor_output(i),
                        output.value[i] if output.value.dim() > 1 else output.value,
                    )

            # Send actions to workers
            for env_idx in running_envs:
                conn_parents[env_idx].send((Command.STEP, env_to_action[env_idx]))

            # Receive new states
            new_worker_datas = {}
            for env_idx in running_envs:
                new_worker_datas[env_idx] = conn_parents[env_idx].recv()

            # Store transitions
            for i, env_idx in enumerate(running_envs):
                actor_out, value = env_to_output[env_idx]
                reward = new_worker_datas[env_idx].reward

                # Re-encode for storage (or store view_game_state and encode later)
                x_game_state_single = encode_batch_view_game_state(
                    [view_game_states[active_envs.index(env_idx)]], device
                )

                transition = Transition(
                    x_game_state=x_game_state_single,
                    fsm=fsms[active_envs.index(env_idx)],
                    action_type=actor_out.action_type,
                    action_type_log_prob=actor_out.action_type_log_prob,
                    secondary_index=actor_out.secondary_index,
                    secondary_log_prob=actor_out.secondary_log_prob,
                    value=value,
                    reward=reward,
                )
                trajectories[env_idx].transitions.append(transition)

            # Update worker data
            for env_idx in running_envs:
                worker_datas[env_idx] = new_worker_datas[env_idx]

    model.train()
    return trajectories


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

    # Add terminal value of 0
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
    all_x_game_states = []
    all_fsms = []
    all_action_types = []
    all_action_type_indices = []
    all_action_type_log_probs = []
    all_secondary_indices = []
    all_secondary_log_probs = []
    all_values = []
    all_returns = []
    all_advantages = []

    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue

        # Compute returns and advantages
        rewards = [t.reward for t in trajectory.transitions]
        values = [t.value for t in trajectory.transitions]
        returns, advantages = _compute_gae(rewards, values, gamma, lam, device)

        for i, trans in enumerate(trajectory.transitions):
            all_x_game_states.append(trans.x_game_state)
            all_fsms.append(trans.fsm)
            all_action_types.append(trans.action_type)

            # Compute action type index within the route
            route = FSM_ROUTING[trans.fsm]
            action_type_idx = route.action_types.index(trans.action_type)
            all_action_type_indices.append(action_type_idx)

            # Log probs (0 if forced/none)
            all_action_type_log_probs.append(
                trans.action_type_log_prob.item()
                if trans.action_type_log_prob is not None
                else 0.0
            )
            all_secondary_indices.append(
                trans.secondary_index if trans.secondary_index is not None else -1
            )
            all_secondary_log_probs.append(
                trans.secondary_log_prob.item() if trans.secondary_log_prob is not None else 0.0
            )

            all_values.append(trans.value)
            all_returns.append(returns[i])
            all_advantages.append(advantages[i])

    # Convert to tensors
    batch = TrajectoryBatch(
        x_game_states=all_x_game_states,
        fsms=all_fsms,
        action_types=all_action_types,
        action_type_indices=torch.tensor(all_action_type_indices, dtype=torch.long, device=device),
        action_type_log_probs=torch.tensor(
            all_action_type_log_probs, dtype=torch.float32, device=device
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


def _minibatch_indices(total: int, minibatch_size: int) -> Iterator[list[int]]:
    """Generate shuffled minibatch indices."""
    indices = list(range(total))
    random.shuffle(indices)

    for i in range(0, total, minibatch_size):
        yield indices[i : i + minibatch_size]


def _recompute_log_probs(
    model: ActorCritic,
    x_game_state: XGameState,
    fsm: FSM,
    action_type: ActionType,
    action_type_idx: int,
    secondary_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute log probs and entropy for a single sample with current policy.

    Returns (total_log_prob, total_entropy, value)
    """
    route = FSM_ROUTING[fsm]

    # Forward through core
    core_out = model.core(x_game_state)

    # Action type head
    if route.is_forced:
        action_type_log_prob = torch.tensor(0.0, device=device)
        action_type_entropy = torch.tensor(0.0, device=device)
    else:
        # Need to get mask - this is a simplification
        # In practice, store masks in trajectory or recompute from view_game_state
        action_type_mask = torch.ones(1, len(route.action_types), dtype=torch.bool, device=device)
        action_type_out = model.head_action_type(core_out.x_global, action_type_mask, sample=False)

        action_type_dist = torch.distributions.Categorical(logits=action_type_out.logits)
        action_type_log_prob = action_type_dist.log_prob(
            torch.tensor([action_type_idx], device=device)
        )[0]
        action_type_entropy = action_type_dist.entropy()[0]

    # Secondary head
    secondary_head_type = get_secondary_head_type(fsm, action_type)
    if secondary_head_type is None or secondary_idx < 0:
        secondary_log_prob = torch.tensor(0.0, device=device)
        secondary_entropy = torch.tensor(0.0, device=device)
    else:
        if secondary_head_type == HeadType.MAP_SELECT:
            # Map head has different interface - uses map encoding, not entities
            secondary_mask = torch.ones(1, MAP_WIDTH, dtype=torch.bool, device=device)
            secondary_out = model.head_map_select(
                core_out.x_map, core_out.x_global, secondary_mask, sample=False
            )
        else:
            x_entities = model._get_entities_for_head(secondary_head_type, core_out)
            secondary_mask = torch.ones(1, x_entities.shape[1], dtype=torch.bool, device=device)
            head = model._secondary_heads[secondary_head_type]
            secondary_out = head(x_entities, core_out.x_global, secondary_mask, sample=False)

        secondary_dist = torch.distributions.Categorical(logits=secondary_out.logits)
        secondary_log_prob = secondary_dist.log_prob(torch.tensor([secondary_idx], device=device))[
            0
        ]
        secondary_entropy = secondary_dist.entropy()[0]

    # Value
    value = model.head_value(core_out.x_global)

    total_log_prob = action_type_log_prob + secondary_log_prob
    total_entropy = action_type_entropy + secondary_entropy

    return total_log_prob, total_entropy, value


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
        for minibatch_idxs in _minibatch_indices(len(batch), minibatch_size):
            # Collect minibatch data
            log_probs_new = []
            entropies = []
            values_new = []

            for idx in minibatch_idxs:
                log_prob, entropy, value = _recompute_log_probs(
                    model,
                    batch.x_game_states[idx],
                    batch.fsms[idx],
                    batch.action_types[idx],
                    batch.action_type_indices[idx].item(),
                    batch.secondary_indices[idx].item(),
                    device,
                )
                log_probs_new.append(log_prob)
                entropies.append(entropy)
                values_new.append(value)

            log_probs_new = torch.stack(log_probs_new)
            entropies = torch.stack(entropies)
            values_new = torch.cat(values_new, dim=0)

            # Old log probs (combined)
            log_probs_old = (
                batch.action_type_log_probs[minibatch_idxs]
                + batch.secondary_log_probs[minibatch_idxs]
            )

            # Advantages and returns
            advantages = torch.squeeze(batch.advantages[minibatch_idxs])
            returns = batch.returns[minibatch_idxs]
            values_old = batch.values[minibatch_idxs]

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

                # Log trajectory stats
                total_reward = sum(
                    sum(t.reward for t in traj.transitions) for traj in trajectories
                )
                avg_length = sum(len(traj) for traj in trajectories) / len(trajectories)
                writer.add_scalar("Trajectory/total_reward", total_reward, episode)
                writer.add_scalar("Trajectory/avg_length", avg_length, episode)

            # Save
            if episode % save_every == 0:
                torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    finally:
        # Clean up workers
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
