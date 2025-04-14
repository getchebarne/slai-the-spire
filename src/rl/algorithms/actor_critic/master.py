import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic.worker import Command
from src.rl.algorithms.actor_critic.worker import WorkerData
from src.rl.algorithms.actor_critic.worker import worker
from src.rl.encoding import unpack_combat_view_encoding
from src.rl.evaluation import run_all_evals
from src.rl.models.actor_critic import ActorCritic
from src.rl.policies import PolicySoftmax
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


@dataclass
class Trajectory:
    states: list[torch.Tensor] = field(default_factory=list)
    valid_action_masks: list[torch.Tensor] = field(default_factory=list)
    action_idxs: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class Batch:
    states: torch.Tensor
    valid_action_masks: torch.Tensor
    action_idxs: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __post_init__(self):
        # Assert all tensor are bidimensional (batch and data dimenson)
        assert (
            2
            == self.states.ndim
            == self.valid_action_masks.ndim
            == self.action_idxs.ndim
            == self.log_probs.ndim
            == self.values.ndim
            == self.returns.ndim
            == self.advantages.ndim
        )

        # Assert all tensors have the same number of samples
        assert (
            self.states.shape[0]
            == self.valid_action_masks.shape[0]
            == self.action_idxs.shape[0]
            == self.log_probs.shape[0]
            == self.values.shape[0]
            == self.returns.shape[0]
            == self.advantages.shape[0]
        )

    def __len__(self) -> int:
        return self.states.shape[0]


def _run_episodes(
    model: ActorCritic, conn_parents: list[Connection], device: torch.device
) -> list[Trajectory]:
    num_env = len(conn_parents)

    # Reset environments
    for conn_parent in conn_parents:
        conn_parent.send((Command.RESET, None))

    # Get initial data from workers
    worker_datas: list[WorkerData] = [conn_parent.recv() for conn_parent in conn_parents]

    # Initialize empty trajectories
    trajectories = [Trajectory() for _ in range(num_env)]

    # Create a list to track the indexes of environments that are still running
    idx_env_running = list(range(num_env))

    # Loop will break when all environments reach a terminal state
    while True:
        combat_view_encodings = []
        valid_action_masks = []
        rewards = []
        idx_env_terminal = []
        for idx_env, worker_data in zip(idx_env_running, worker_datas):
            if worker_data.game_over_flag:
                idx_env_terminal.append(idx_env)

                continue

            combat_view_encodings.append(worker_data.combat_view_encoding)
            valid_action_masks.append(worker_data.valid_action_mask)
            rewards.append(worker_data.reward)

        # Remove environments that have reached a terminal state
        for idx_env in idx_env_terminal:
            idx_env_running.remove(idx_env)

        # If all environments have ended, break the loop
        if not idx_env_running:
            break

        # Else, get action indexes for each running environment. Concatenate combat view encodings
        # and create valid action masks tensor
        combat_view_encodings = torch.cat(combat_view_encodings, dim=0)
        valid_action_masks = torch.tensor(valid_action_masks, dtype=torch.bool, device=device)

        # Forward pass
        x_probs, x_values = model(
            *unpack_combat_view_encoding(combat_view_encodings).as_tuple(),
            x_valid_action_mask=valid_action_masks,
        )

        # Sample action indexes for each environment
        dist = torch.distributions.Categorical(probs=x_probs)
        action_idxs = dist.sample().view(len(idx_env_running), 1)

        # Send action indexes to their corresponding workers
        for idx_batch, idx_env in enumerate(idx_env_running):
            conn_parents[idx_env].send((Command.STEP, action_idxs[idx_batch : idx_batch + 1]))

        # Overwrite previous `worker_datas` with new data from workers
        worker_datas: list[WorkerData] = [
            conn_parents[idx_env].recv() for idx_env in idx_env_running
        ]

        # Store the transition information
        log_probs = dist.log_prob(torch.flatten(action_idxs)).view(len(idx_env_running), 1)
        for idx_batch, idx_env in enumerate(idx_env_running):
            trajectory = trajectories[idx_env]

            trajectory.states.append(combat_view_encodings[idx_batch : idx_batch + 1])
            trajectory.valid_action_masks.append(valid_action_masks[idx_batch : idx_batch + 1])
            trajectory.action_idxs.append(action_idxs[idx_batch : idx_batch + 1])
            trajectory.log_probs.append(log_probs[idx_batch : idx_batch + 1])
            trajectory.values.append(x_values[idx_batch : idx_batch + 1])
            trajectory.rewards.append(worker_datas[idx_batch].reward)

    return trajectories


def _compute_returns_and_advantages_gae(
    trajectory: Trajectory, gamma: float, lam: float, device: torch.device
) -> tuple[deque[torch.Tensor], deque[torch.Tensor]]:
    # Use `deque` here so that appending to the beginning of the list is O(1)
    returns = deque()
    advantages = deque()

    # Create dummy next-state value equal to zero because all trajectories are episodic, i.e, the
    # last state is always terminal
    values_next = trajectory.values[1:] + [
        torch.tensor([[0.0]], dtype=torch.float32, device=device)
    ]

    # Initialize generalized advantage estimator to zero
    gae = 0
    for reward, value, value_next in zip(
        reversed(trajectory.rewards), reversed(trajectory.values), reversed(values_next)
    ):
        td_error = reward + gamma * value_next.item() - value.item()
        gae = td_error + gamma * lam * gae

        returns.appendleft(gae + value.item())
        advantages.appendleft(gae)

    return returns, advantages


def _create_batch(
    trajectories: list[Trajectory],
    gamma: float,
    lam: float,
    device: torch.device,
) -> Batch:
    # Initialize lists to store all TODO: improve comment
    all_states = []
    all_valid_action_masks = []
    all_action_idxs = []
    all_log_probs = []
    all_values = []
    all_returns = []
    all_advantages = []

    for trajectory in trajectories:
        returns, advantages = _compute_returns_and_advantages_gae(trajectory, gamma, lam, device)

        # Extend lists TODO: improve comment
        all_states.extend(trajectory.states)
        all_valid_action_masks.extend(trajectory.valid_action_masks)
        all_action_idxs.extend(trajectory.action_idxs)
        all_log_probs.extend(trajectory.log_probs)
        all_values.extend(trajectory.values)
        all_returns.extend(returns)
        all_advantages.extend(advantages)

    # Create the batch
    batch = Batch(
        torch.cat(all_states, dim=0),
        torch.cat(all_valid_action_masks, dim=0),
        torch.cat(all_action_idxs, dim=0),
        torch.cat(all_log_probs, dim=0).detach(),
        torch.cat(all_values, dim=0).detach(),  # TODO: parametrize detach if clipping value loss
        # New tensors for returns and advantages
        torch.tensor([all_returns], dtype=torch.float32, device=device).view(-1, 1),
        torch.tensor([all_advantages], dtype=torch.float32, device=device).view(-1, 1),
    )

    # Normalize advantages
    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std() + 1e-8
    )

    return batch


def _dataloader(total: int, minibatch_size: int) -> Iterator[list[int]]:
    indices = list(range(total))
    random.shuffle(indices)
    num_full_batches = total // minibatch_size

    for i in range(num_full_batches):
        start = i * minibatch_size
        yield indices[start : start + minibatch_size]


def _update_model_ppo(
    model: ActorCritic,
    trajectories: list[Trajectory],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    gamma: float,
    lam: float,
    clip_eps: float,
    clip_value_loss: bool,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> None:
    # Create batch
    batch = _create_batch(trajectories, gamma, lam, device)

    # PPO Training loop
    total_loss_policy = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_loss_value = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_loss_entropy = torch.tensor([0.0], dtype=torch.float32, device=device)
    for _ in range(num_epochs):
        minibatch_num = 0
        epoch_loss_policy = torch.tensor([0.0], dtype=torch.float32, device=device)
        epoch_loss_value = torch.tensor([0.0], dtype=torch.float32, device=device)
        epoch_loss_entropy = torch.tensor([0.0], dtype=torch.float32, device=device)
        for minibatch_idxs in _dataloader(len(batch), minibatch_size):
            minibatch_num += 1

            # Get minibatch
            minibatch_states = batch.states[minibatch_idxs]
            minibatch_valid_action_masks = batch.valid_action_masks[minibatch_idxs]
            minibatch_action_idxs = batch.action_idxs[minibatch_idxs]
            minibatch_log_probs = batch.log_probs[minibatch_idxs]
            minibatch_values = batch.values[minibatch_idxs]
            minibatch_returns = batch.returns[minibatch_idxs]
            minibatch_advantages = batch.advantages[minibatch_idxs]

            # Re-compute log probs and values with current policy
            x_probs, x_values = model(
                *unpack_combat_view_encoding(minibatch_states).as_tuple(),
                x_valid_action_mask=minibatch_valid_action_masks,
            )
            dist = torch.distributions.Categorical(probs=x_probs)
            minibatch_log_probs_new = dist.log_prob(torch.flatten(minibatch_action_idxs)).view(
                -1, 1
            )
            entropies = dist.entropy()

            # Policy loss: PPO clipped surrogate
            ratio = torch.exp(minibatch_log_probs_new - minibatch_log_probs)
            surr_1 = ratio * minibatch_advantages
            surr_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * minibatch_advantages
            loss_policy = -torch.mean(torch.min(surr_1, surr_2))

            # Value loss: optional clipping
            if clip_value_loss:
                value_pred_clipped = minibatch_values + (x_values - minibatch_values).clamp(
                    -clip_eps, clip_eps
                )
                loss_value_unclipped = (x_values - minibatch_returns).pow(2)
                loss_value_clipped = (value_pred_clipped - minibatch_returns).pow(2)
                loss_value = torch.mean(torch.max(loss_value_unclipped, loss_value_clipped)) / 2
            else:
                loss_value = F.mse_loss(x_values, minibatch_returns)

            # Entropy bonus
            loss_entropy = -torch.mean(entropies)

            # Total loss
            loss_total = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            # Backprop
            optimizer.zero_grad()
            loss_total.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Take step
            optimizer.step()

            # Accumulate losses
            epoch_loss_policy += loss_policy
            epoch_loss_value += loss_value
            epoch_loss_entropy += loss_entropy

        # Average losses across the epoch and add them to the total
        total_loss_policy += epoch_loss_policy / minibatch_num
        total_loss_value += epoch_loss_value / minibatch_num
        total_loss_entropy += epoch_loss_entropy / minibatch_num

    return (
        total_loss_policy.item() / num_epochs,
        total_loss_value.item() / num_epochs,
        total_loss_entropy.item() / num_epochs,
    )


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
    clip_value_loss: float,
    gamma: float,
    lam: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
    num_eval: int,
    num_envs: int,
    device: torch.device,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Send model to device
    model.to(device)

    # Initialize policy
    policy = PolicySoftmax(model, device, greedy=True)

    # Initialize connections
    conn_parents, conn_workers = zip(*[Pipe() for _ in range(num_envs)])

    # Start worker processes
    processes = [
        Process(target=worker, args=(conn_worker, device)) for conn_worker in conn_workers
    ]
    for process in processes:
        process.start()

    for num_episode in range(num_episodes):
        coef_entropy = coefs_entropy[num_episode]

        # Run games, get trajectories
        trajectories = _run_episodes(model, conn_parents, device)
        loss_policy, loss_value, loss_entropy = _update_model_ppo(
            model,
            trajectories,
            optimizer,
            num_epochs,
            minibatch_size,
            gamma,
            lam,
            clip_eps,
            clip_value_loss,
            coef_value,
            coef_entropy,
            max_grad_norm,
            device,
        )

        if (num_episode % log_every) == 0:
            writer.add_scalar("Loss/policy", loss_policy, num_episode)
            writer.add_scalar("Loss/value", loss_value, num_episode)
            writer.add_scalar("Entropy/coef.", coef_entropy, num_episode)
            writer.add_scalar("Entropy/value", -1 * loss_entropy, num_episode)

            eval_results = run_all_evals(policy, num_eval)
            for eval_name, eval_values in eval_results.items():
                writer.add_scalar(
                    f"Evaluation/{eval_name}", np.mean(eval_values).item(), num_episode
                )

        # Save model
        if (num_episode % save_every) == 0:
            torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")


def _get_coefs_entropy(num_episodes: int, elbow: int, max_: float, min_: float) -> list[float]:
    coefs_entropy = []
    slope = (min_ - max_) / elbow
    offset = max_
    for num_episode in range(num_episodes):
        if num_episode <= elbow:
            coef_entropy = slope * num_episode + offset
        else:
            coef_entropy = min_

        coefs_entropy.append(coef_entropy)

    return coefs_entropy


if __name__ == "__main__":
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)

    # Model
    model = ActorCritic(**config["model"])

    # Optimizer
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])

    # Copy config to experiment directory
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    # Get entropy coefficients
    coefs_entropy = _get_coefs_entropy(
        config["num_episodes"],
        config["coef_entropy_elbow"],
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )

    # Start training
    print(f"{config['exp_name']=}")
    train(
        config["exp_name"],
        config["num_episodes"],
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
        config["num_eval"],
        config["num_envs"],
        torch.device("cpu"),
    )
