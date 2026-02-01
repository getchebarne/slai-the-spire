"""
Single-threaded A2C training script (simpler, for debugging).

This is a simpler alternative to the parallel master/worker setup.
Useful for debugging and understanding the training loop.
"""

import os
import shutil
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.game.action import Action
from src.game.action import ActionType
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.state import get_view_game_state
from src.rl.action_space import FSM_ROUTING
from src.rl.action_space import HeadType
from src.rl.action_space import get_secondary_head_type
from src.rl.action_space.masks import get_valid_mask_batch
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import encode_batch_view_game_state
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


def _get_masks(view_game_state, fsm: FSM, device: torch.device) -> dict[HeadType, torch.Tensor]:
    """Get all relevant masks for a single state."""
    masks = {
        HeadType.ACTION_TYPE: get_valid_mask_batch(
            HeadType.ACTION_TYPE, [view_game_state], device
        )
    }

    # Add masks for potential secondary heads
    route = FSM_ROUTING[fsm]
    for action_type in route.action_types:
        secondary_head = get_secondary_head_type(fsm, action_type)
        if secondary_head is not None and secondary_head not in masks:
            masks[secondary_head] = get_valid_mask_batch(
                secondary_head, [view_game_state], device
            )

    return masks


def _play_episode(model: ActorCritic, device: torch.device) -> tuple[EpisodeResult, int]:
    """
    Play a single episode and collect trajectory.

    Returns (EpisodeResult, final_floor)
    """
    game_state = create_game_state(ASCENSION_LEVEL)
    initialize_game_state(game_state)

    result = EpisodeResult()

    while game_state.fsm != FSM.GAME_OVER:
        view_game_state = get_view_game_state(game_state)
        fsm = game_state.fsm

        # Encode state
        x_game_state = encode_batch_view_game_state([view_game_state], device)

        # Get masks
        masks = _get_masks(view_game_state, fsm, device)

        # Forward pass
        output = model(x_game_state, fsm, masks, sample=True)

        # Build action
        action = output.actor.to_action()

        # Execute action
        step(game_state, action)

        # Get reward
        view_game_state_next = get_view_game_state(game_state)
        game_over = game_state.fsm == FSM.GAME_OVER
        reward = compute_reward(view_game_state, view_game_state_next, game_over)

        # Store transition
        result.log_probs.append(torch.unsqueeze(output.actor.total_log_prob, 0))
        result.values.append(output.value)
        result.rewards.append(reward)

        # Compute entropy (simplified - just from value, actual entropy needs distribution)
        # In practice you'd want to compute this from the action distributions
        result.entropies.append(torch.tensor([0.01], device=device))  # Placeholder

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
