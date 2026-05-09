"""
Single-threaded A2C training script (simpler, for debugging).

This is a simpler alternative to the parallel master/worker setup.
Useful for debugging and understanding the training loop.
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.action_space.masks import get_mask_batch
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.env_wrapper import EnvWrapper
from src.rl.models import ActorCritic
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)


def _play_episode(model: ActorCritic, device: torch.device) -> tuple[EpisodeResult, int]:
    """
    Play a single episode and collect trajectory.

    Returns (EpisodeResult, final_floor)
    """
    wrapper = EnvWrapper(ascension=ASCENSION_LEVEL)
    wrapper.reset(seed=random.randint(0, 2**31 - 1))

    result = EpisodeResult()
    terminated = False
    view_game_state_next = wrapper.obs

    while not terminated:
        view_game_state = wrapper.obs

        # Encode state
        x_game_state = encode_batch_view_game_state([view_game_state], device)

        # Get masks (route.py reads wrapper.is_awaiting_target so pass the wrapper)
        mask_batch = get_mask_batch([wrapper], device)

        # Forward pass
        output = model.forward_single(x_game_state, mask_batch, sample=True)

        # Build action (may be a slai.Action.* or a buffering marker)
        action = output.to_action()

        # Execute action
        view_game_state_next, _engine_reward, terminated, _trunc, _info = wrapper.step(action)

        # Get reward
        reward = compute_reward(view_game_state, view_game_state_next, terminated)

        # Store transition
        result.log_probs.append(torch.unsqueeze(output.log_prob, 0))
        result.values.append(output.value.unsqueeze(0))
        result.rewards.append(reward)

        # Compute entropy (simplified placeholder)
        result.entropies.append(torch.tensor([0.01], device=device))

    final_floor = view_game_state_next.map.y_current or 0
    return result, final_floor


def _update_model(
    result: EpisodeResult,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """A2C update step."""
    # Compute discounted returns
    returns = deque()
    advantages = deque()

    return_disc = 0.0
    for reward, value in zip(reversed(result.rewards), reversed(result.values)):
        return_disc = reward + gamma * return_disc
        advantage = return_disc - value.item()
        returns.appendleft(return_disc)
        advantages.appendleft(advantage)

    returns = torch.tensor(list(returns), dtype=torch.float32, device=device)
    advantages = torch.tensor(list(advantages), dtype=torch.float32, device=device)

    # Stack tensors
    log_probs = torch.cat(result.log_probs)
    values = torch.squeeze(torch.cat(result.values))
    entropies = torch.cat(result.entropies)

    # Losses
    loss_policy = -torch.mean(log_probs * advantages)
    loss_value = F.mse_loss(values, returns)
    loss_entropy = -torch.mean(entropies)
    loss_total = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

    # Backward
    optimizer.zero_grad()
    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss_policy.item(), loss_value.item(), loss_entropy.item()


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
    gamma: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
    device: torch.device,
) -> None:
    """Main training loop."""
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)

    for episode in range(num_episodes):
        coef_entropy = coefs_entropy[episode]

        # Play episode
        result, final_floor = _play_episode(model, device)

        # Update model
        loss_policy, loss_value, loss_entropy = _update_model(
            result,
            model,
            optimizer,
            gamma,
            coef_value,
            coef_entropy,
            max_grad_norm,
            device,
        )

        # Logging
        if episode % log_every == 0:
            print(f"Episode {episode}: floor={final_floor}, policy={loss_policy:.4f}")
            writer.add_scalar("Loss/policy", loss_policy, episode)
            writer.add_scalar("Loss/value", loss_value, episode)
            writer.add_scalar("Entropy/coef", coef_entropy, episode)
            writer.add_scalar("Entropy/value", -loss_entropy, episode)
            writer.add_scalar("Floor", final_floor, episode)

            total_reward = sum(result.rewards)
            writer.add_scalar("Reward/total", total_reward, episode)
            writer.add_scalar("Episode/length", len(result.rewards), episode)

        # Save
        if episode % save_every == 0:
            torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

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

    # Train (single-threaded A2C)
    print(f"Starting training: {config['exp_name']}")
    train(
        config["exp_name"],
        int(config["num_episodes"]),
        config["log_every"],
        config["save_every"],
        model,
        optimizer,
        config["gamma"],
        config["coef_value"],
        coefs_entropy,
        config["max_grad_norm"],
        torch.device("cpu"),
    )
