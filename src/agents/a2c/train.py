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
from src.agents.a2c.evaluation import evaluate_blunder
from src.agents.a2c.evaluation import evaluate_lethal
from src.agents.a2c.models.actor import Actor
from src.agents.a2c.models.actor import action_idx_to_action
from src.agents.a2c.models.actor import get_valid_action_mask
from src.agents.a2c.models.critic import Critic
from src.agents.a2c.reward import compute_reward
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


def _play_episode(
    model_actor: Actor, model_critic: Critic, device: torch.device
) -> tuple[CombatState, EpisodeResult]:
    # Get new game TODO: improve this
    cs = create_combat_state()
    start_combat(cs)

    episode_result = EpisodeResult()
    while not is_game_over(cs.entity_manager):
        # Get combat view, encode it, and get valid action mask
        combat_view_t = view_combat(cs)
        combat_view_t_encoded, index_mapping = encode_combat_view(combat_view_t, device)
        valid_action_mask_t = get_valid_action_mask(combat_view_t)

        # Get action probabilities and state value
        x_prob = model_actor(
            combat_view_t_encoded,
            torch.tensor(valid_action_mask_t, dtype=torch.bool, device=device),
        )
        x_value = model_critic(combat_view_t_encoded)

        # Sample action from the action-selection distribution
        dist = torch.distributions.Categorical(x_prob)
        action_idx = dist.sample()

        # Game step
        action = action_idx_to_action(action_idx.item(), combat_view_t, index_mapping)
        step(cs, action)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_tp1 = view_combat(cs)
        game_over_flag = is_game_over(cs.entity_manager)
        reward = compute_reward(combat_view_t, combat_view_tp1, game_over_flag)

        # Store the transition information
        episode_result.log_probs.append(dist.log_prob(action_idx).unsqueeze(0))
        episode_result.values.append(x_value)
        episode_result.rewards.append(reward)
        episode_result.entropies.append(dist.entropy().unsqueeze(0))

    return cs, episode_result


def _update_model(
    episode_result: EpisodeResult,
    model_actor: Actor,
    model_critic: Critic,
    optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    gamma: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float]:
    # Calculate TD targets and advantages
    td_targets = deque()
    advantages = deque()

    # For the last step, there's no next state, so we use just the reward
    next_value = 0
    for reward, value_t, value_tp1 in zip(
        reversed(episode_result.rewards),
        reversed(episode_result.values),
        reversed(
            episode_result.values[1:]
            + [torch.tensor([next_value], dtype=torch.float32, device=device)]
        ),
    ):
        # Calculate TD target: r + γV(s')
        td_target = reward + gamma * value_tp1.item()
        td_targets.appendleft(td_target)

        # Calculate advantage: TD target - V(s)
        advantage = td_target - value_t.item()
        advantages.appendleft(advantage)

    td_targets = torch.tensor(td_targets, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Convert lists of 1-dimensional tensors to single len-episode-tensors
    log_probs = torch.cat(episode_result.log_probs)
    values = torch.cat(episode_result.values)
    entropies = torch.cat(episode_result.entropies)

    # Calculate losses
    loss_policy = -1 * torch.mean(log_probs * advantages)
    loss_entropy = -1 * torch.mean(entropies)
    loss_value = F.mse_loss(values.squeeze(), td_targets)

    # Calculate gradients
    loss_policy_entropy = loss_policy + coef_entropy * loss_entropy

    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    loss_policy_entropy.backward()
    loss_value.backward()

    # Apply clipping
    nn.utils.clip_grad_norm_(model_actor.parameters(), max_grad_norm)
    nn.utils.clip_grad_norm_(model_critic.parameters(), max_grad_norm)

    # Take optimization step
    optimizer_actor.step()
    optimizer_critic.step()

    # Return losses for logging
    return loss_policy.item(), loss_value.item(), loss_entropy.item()


def _evaluate_agent(model_actor: Actor, device: torch.device) -> int:
    # Get new game TODO: improve this
    cs = create_combat_state()
    start_combat(cs)

    while not is_game_over(cs.entity_manager):
        # Get combat view
        combat_view = view_combat(cs)
        combat_view_encoded, index_mapping = encode_combat_view(combat_view, device)
        valid_action_mask = get_valid_action_mask(combat_view)

        # Get action from agent
        x_prob = model_actor(
            combat_view_encoded,
            torch.tensor(valid_action_mask, dtype=torch.float32, device=device),
        )

        # Game step
        action_idx = torch.argmax(x_prob).item()
        action = action_idx_to_action(action_idx, combat_view, index_mapping)

        step(cs, action)

    # Return final health
    return view_combat(cs).character.health_current


def train(
    exp_name: str,
    num_episodes: int,
    log_every: int,
    eval_every: int,
    save_every: int,
    model_actor: Actor,
    model_critic: Critic,
    optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    device: torch.device,
    gamma: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Send models to device
    model_actor.to(device)
    model_critic.to(device)

    # Train
    for num_episode in range(num_episodes):
        # Play episode, get trajectory
        cs, episode_result = _play_episode(model_actor, model_critic, device)

        # Fit the models
        coef_entropy = coefs_entropy[num_episode]
        loss_policy, loss_value, loss_entropy = _update_model(
            episode_result,
            model_actor,
            model_critic,
            optimizer_actor,
            optimizer_critic,
            gamma,
            coef_entropy,
            max_grad_norm,
            device,
        )
        combat_view_end = view_combat(cs)

        if (num_episode % log_every) == 0:
            writer.add_scalar("Loss/policy", loss_policy, num_episode)
            writer.add_scalar("Loss/value", loss_value, num_episode)
            writer.add_scalar(
                "Training/Health", combat_view_end.character.health_current, num_episode
            )
            writer.add_scalar("Entropy/coef.", coef_entropy, num_episode)
            writer.add_scalar("Entropy/value", -1 * loss_entropy, num_episode)

        # Evaluate agent
        if (num_episode % eval_every) == 0:
            hp_final = _evaluate_agent(model_actor, device)

            # TODO: parametrize
            num_eval = 10
            blunder = []
            lethal = []
            for _ in range(num_eval):
                blunder.append(evaluate_blunder(model_actor, device))
                lethal.append(evaluate_lethal(model_actor, device))

            writer.add_scalar("Evaluation/Health", hp_final, num_episode)
            writer.add_scalar("Scenario/Blunder", sum(blunder) / len(blunder), num_episode)
            writer.add_scalar("Scenario/Lethal", sum(lethal) / len(lethal), num_episode)

        # Save models
        if (num_episode % save_every) == 0:
            torch.save(model_actor.state_dict(), f"experiments/{exp_name}/model_actor.pth")
            torch.save(model_critic.state_dict(), f"experiments/{exp_name}/model_critic.pth")


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
def _init_optimizer(name: str, model: Actor | Critic, **kwargs) -> torch.optim.Optimizer:
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

    # Models
    model_actor = Actor(config["model_actor"]["dim_card"])
    model_critic = Critic(config["model_critic"]["dim_card"])

    # Optimizers
    optimizer_actor = _init_optimizer(
        config["optimizer_actor"]["name"], model_actor, **config["optimizer_actor"]["kwargs"]
    )
    optimizer_critic = _init_optimizer(
        config["optimizer_critic"]["name"], model_critic, **config["optimizer_critic"]["kwargs"]
    )

    # Copy config to experiment directory
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy("src/agents/a2c/config.yml", f"experiments/{config['exp_name']}/config.yml")

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
        config["eval_every"],
        config["save_every"],
        model_actor,
        model_critic,
        optimizer_actor,
        optimizer_critic,
        torch.device("cpu"),
        config["gamma"],
        coefs_entropy,
        config["max_grad_norm"],
    )
