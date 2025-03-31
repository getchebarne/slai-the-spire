import os
import shutil
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.a2c.encode import encode_combat_view
from src.agents.a2c.model import ActorCritic
from src.agents.a2c.reward import compute_reward
from src.agents.a2c.utils import action_idx_to_action
from src.agents.a2c.utils import get_valid_action_mask
from src.game.combat.create import create_combat_state
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.state import CombatState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


@dataclass
class EpisodeResult:
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)


def _play_episode(model: ActorCritic, device: torch.device) -> tuple[CombatState, EpisodeResult]:
    # Get new game TODO: improve this
    cs = create_combat_state()
    start_combat(cs)

    episode_result = EpisodeResult()
    while not is_game_over(cs.entity_manager):
        # Get combat view, encode it, and get valid action mask
        combat_view_t = view_combat(cs)
        combat_view_t_encoded = encode_combat_view(combat_view_t, device)
        valid_action_mask_t = get_valid_action_mask(combat_view_t)

        # Get action probabilities and state value
        probs, value = model(
            combat_view_t_encoded,
            torch.tensor(valid_action_mask_t, dtype=torch.bool, device=device),
        )

        # Sample action from the action-selection distribution
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        # Game step
        action = action_idx_to_action(action_idx.item(), combat_view_t)
        step(cs, action)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_tp1 = view_combat(cs)
        game_over_flag = is_game_over(cs.entity_manager)
        reward = compute_reward(combat_view_t, combat_view_tp1, game_over_flag)

        # Store the transition information
        episode_result.log_probs.append(dist.log_prob(action_idx).unsqueeze(0))
        episode_result.values.append(value)
        episode_result.rewards.append(reward)
        episode_result.entropies.append(dist.entropy().unsqueeze(0))

    return cs, episode_result


def _update_model(
    episode_result: EpisodeResult,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> float:
    # Compute returns and advantages. Use `deque` so that inserting to the left is O(1)
    ret_discs = deque()
    advantages = deque()

    ret_disc = 0
    for reward, value in zip(reversed(episode_result.rewards), reversed(episode_result.values)):
        # Calculate discounted return
        ret_disc = reward + gamma * ret_disc
        ret_discs.appendleft(ret_disc)

        # Calculate advantage
        advantage = ret_disc - value.item()  # TODO: detaching `value` from compute graph?
        advantages.appendleft(advantage)

    ret_discs = torch.tensor(ret_discs, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert lists of 1-dimensional tensors to single len-episode-tensors
    log_probs = torch.cat(episode_result.log_probs)
    values = torch.cat(episode_result.values)
    entropies = torch.cat(episode_result.entropies)

    # Calculate losses
    policy_loss = -1 * torch.mean(log_probs * advantages)
    value_loss = F.mse_loss(values.squeeze(), ret_discs)
    entropy_loss = -1 * torch.mean(entropies)

    # Total loss
    loss = policy_loss + coef_value * value_loss + coef_entropy * entropy_loss

    # Calculate gradients. Clip them if necessary
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Take optimization step and return episode loss
    optimizer.step()

    return loss.item()


def _evaluate_agent(model: ActorCritic, device: torch.device) -> int:
    # Get new game TODO: improve this
    cs = create_combat_state()
    start_combat(cs)

    while not is_game_over(cs.entity_manager):
        # Get combat view
        combat_view = view_combat(cs)
        valid_action_mask = get_valid_action_mask(combat_view)
        combat_view_encoded = encode_combat_view(combat_view, device)

        # Get action from agent
        probs, _ = model(
            combat_view_encoded,
            torch.tensor(valid_action_mask, dtype=torch.float32, device=device),
        )

        # Game step
        action_idx = torch.argmax(probs).item()
        action = action_idx_to_action(action_idx, combat_view)

        step(cs, action)

    # Return final health
    return view_combat(cs).character.health_current


def train(
    exp_name: str,
    num_episodes: int,
    log_every: int,
    eval_every: int,
    save_every: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gamma: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Send model to device TODO: move
    model.to(device)

    # Train
    for num_episode in range(num_episodes):
        cs, episode_result = _play_episode(model, device)
        coef_entropy = coefs_entropy[num_episode]
        loss = _update_model(
            episode_result, optimizer, gamma, coef_value, coef_entropy, max_grad_norm, device
        )
        combat_view_end = view_combat(cs)

        if (num_episode % log_every) == 0:
            writer.add_scalar("Loss", loss, num_episode)
            writer.add_scalar("Final HP", combat_view_end.character.health_current, num_episode)
            writer.add_scalar("Entropy coef.", coef_entropy, num_episode)

        # Evaluate agent
        if (num_episode % eval_every) == 0:
            hp_final = _evaluate_agent(model, device)
            writer.add_scalar("Eval/Final HP", hp_final, num_episode)

        # Save model
        if (num_episode % save_every) == 0:
            torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")


# TODO: fix path
# TODO: parse values accordingly
def _load_config(config_path: str = "src/agents/a2c/config.yml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    config["num_episodes"] = int(config["num_episodes"])
    config["coef_entropy_elbow"] = int(config["coef_entropy_elbow"])

    return config


# TODO: use **kwargs, improve signature
def _init_optimizer(name: str, model: ActorCritic, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, name)(**kwargs, params=model.parameters())


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
    config = _load_config()

    model = ActorCritic(config["model"]["dim_card"], config["model"]["num_heads"])
    optimizer = _init_optimizer(
        config["optimizer"]["name"], model, **config["optimizer"]["kwargs"]
    )

    # Copy config
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy("src/agents/a2c/config.yml", f"experiments/{config['exp_name']}/config.yml")

    coefs_entropy = _get_coefs_entropy(
        config["num_episodes"],
        config["coef_entropy_elbow"],
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )
    train(
        config["exp_name"],
        config["num_episodes"],
        config["log_every"],
        config["eval_every"],
        config["save_every"],
        model,
        optimizer,
        torch.device("cpu"),
        config["gamma"],
        config["coef_value"],
        coefs_entropy,
        config["max_grad_norm"],
    )
