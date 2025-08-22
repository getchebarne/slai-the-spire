import os
import shutil
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.encoding.state import encode_view_game_state
from src.rl.models.actor_critic import ActorCritic
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


_ASCENSION_LEVEL = 1


@dataclass
class EpisodeResult:
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)


def _play_episode(model: ActorCritic, device: torch.device) -> tuple[EpisodeResult, ViewGameState]:
    # Get new game TODO: improve this
    game_state = create_game_state(_ASCENSION_LEVEL)
    initialize_game_state(game_state)

    num_step = 0
    episode_result = EpisodeResult()
    while game_state.fsm != FSM.GAME_OVER:
        # Get combat view, encode it, and get valid action mask
        view_game_state_t = get_view_game_state(game_state)
        encoding_game_state_t = encode_view_game_state(view_game_state_t, device)
        valid_action_mask_t = get_valid_action_mask(view_game_state_t)

        # Get action probabilities and state value
        x_prob, x_value = model(*encoding_game_state_t, valid_action_mask_t)

        # Sample action from the action-selection distribution
        dist = torch.distributions.Categorical(x_prob)
        action_idx = dist.sample()

        # Game step
        action = action_idx_to_action(action_idx.item())
        step(game_state, action)

        # Get new state, new valid actions, game over flag and instant reward
        view_game_state_tp1 = get_view_game_state(game_state)
        game_over_flag = game_state.fsm == FSM.GAME_OVER
        reward = compute_reward(view_game_state_t, view_game_state_tp1, game_over_flag)

        # Store the transition information
        episode_result.log_probs.append(dist.log_prob(action_idx).unsqueeze(0))
        episode_result.values.append(x_value)
        episode_result.rewards.append(reward)
        episode_result.entropies.append(dist.entropy().unsqueeze(0))

        num_step += 1

    return episode_result, view_game_state_tp1


def _update_model(
    episode_result: EpisodeResult,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float]:
    # Initialize empty deques to store discounted returns and advantages. Use `deque` so that
    # inserting to the left is O(1)
    return_discs = deque()
    advantages = deque()

    # Intialize discounted return to zero for terminal state
    return_disc = 0
    for reward, value in zip(reversed(episode_result.rewards), reversed(episode_result.values)):
        # Calculate discounted return & advantage of selected action
        return_disc = reward + gamma * return_disc
        advantage = return_disc - value.item()  # Detached `value` from graph

        # Insert at position 0
        return_discs.appendleft(return_disc)
        advantages.appendleft(advantage)

    # Convert to tensors
    return_discs = torch.tensor(return_discs, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Convert lists of 1-dimensional tensors to single len-episode-tensors
    log_probs = torch.cat(episode_result.log_probs)
    values = torch.cat(episode_result.values)
    entropies = torch.cat(episode_result.entropies)

    # Calculate losses
    loss_policy = -1 * torch.mean(log_probs * advantages)
    loss_value = F.mse_loss(values.squeeze(), return_discs)
    loss_entropy = -1 * torch.mean(entropies)
    loss_total = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

    # Calculate gradients
    optimizer.zero_grad()
    loss_total.backward()

    # Apply clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Take optimization step
    optimizer.step()

    # Return each loss term
    return loss_policy.item(), loss_value.item(), loss_entropy.item()


def train(
    exp_name: str,
    num_episodes: int,
    log_every: int,
    save_every: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gamma: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
    num_eval: int,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Send model to device
    model.to(device)

    # Train
    for num_episode in range(num_episodes):
        print(f"{num_episode=}")

        # Play episode, get trajectory
        episode_result, view_game_state_final = _play_episode(model, device)

        # Fit the models
        coef_entropy = coefs_entropy[num_episode]
        loss_policy, loss_value, loss_entropy = _update_model(
            episode_result,
            model,
            optimizer,
            gamma,
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
            writer.add_scalar("Floor", view_game_state_final.map.y_current, num_episode)

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
        torch.device("cpu"),
        config["gamma"],
        config["coef_value"],
        coefs_entropy,
        config["max_grad_norm"],
        config["num_eval"],
    )
