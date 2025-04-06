import os
import shutil
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.game.combat.create import create_combat_state
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.state import CombatState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat
from src.rl.encoding import encode_combat_view
from src.rl.evaluation import evaluate_blunder
from src.rl.evaluation import evaluate_dagger_throw_vs_strike
from src.rl.evaluation import evaluate_draw_first_w_backflip
from src.rl.evaluation import evaluate_final_hp
from src.rl.evaluation import evaluate_lethal
from src.rl.models.actor_critic import ActorCritic
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask
from src.rl.policies import PolicySoftmax
from src.rl.reward import compute_reward


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
        x_prob, x_value = model(
            combat_view_t_encoded,
            torch.tensor(valid_action_mask_t, dtype=torch.bool, device=device),
        )

        # Sample action from the action-selection distribution
        dist = torch.distributions.Categorical(x_prob)
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
        episode_result.values.append(x_value)
        episode_result.rewards.append(reward)
        episode_result.entropies.append(dist.entropy().unsqueeze(0))

    return cs, episode_result


def _update_model(
    episode_result: EpisodeResult,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
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
    loss_total = loss_policy + coef_entropy * loss_entropy + loss_value  # TODO: re-add coef_value

    # Calculate gradients
    optimizer.zero_grad()
    loss_total.backward()

    # Apply clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Take optimization step
    optimizer.step()

    # Return losses for logging
    return loss_policy.item(), loss_value.item(), loss_entropy.item()


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
    coefs_entropy: list[float],
    max_grad_norm: float,
    num_eval: int,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Send model to device
    model.to(device)

    # Initialize policy
    policy = PolicySoftmax(model, device, greedy=True)

    # Train
    for episode in range(num_episodes):
        # Play episode, get trajectory
        cs, episode_result = _play_episode(model, device)

        # Fit the models
        coef_entropy = coefs_entropy[episode]
        loss_policy, loss_value, loss_entropy = _update_model(
            episode_result, model, optimizer, gamma, coef_entropy, max_grad_norm, device
        )
        combat_view_end = view_combat(cs)

        if (episode % log_every) == 0:
            writer.add_scalar("Loss/policy", loss_policy, episode)
            writer.add_scalar("Loss/value", loss_value, episode)
            writer.add_scalar("Training/Health", combat_view_end.character.health_current, episode)
            writer.add_scalar("Entropy/coef.", coef_entropy, episode)
            writer.add_scalar("Entropy/value", -1 * loss_entropy, episode)

            # TODO: abstract this
            mean_final_hp = (
                sum([evaluate_final_hp(policy, device) for _ in range(num_eval)]) / num_eval
            )
            mean_blunder = (
                sum([evaluate_blunder(policy, device) for _ in range(num_eval)]) / num_eval
            )
            mean_lethal = (
                sum([evaluate_lethal(policy, device) for _ in range(num_eval)]) / num_eval
            )
            mean_draw_first_w_backflip = (
                sum([evaluate_draw_first_w_backflip(policy, device) for _ in range(num_eval)])
                / num_eval
            )
            mean_dagger_throw_vs_strike = (
                sum([evaluate_dagger_throw_vs_strike(policy, device) for _ in range(num_eval)])
                / num_eval
            )

            writer.add_scalar("Evaluation/Final health", mean_final_hp, episode)
            writer.add_scalar("Evaluation/Blunder", mean_blunder, episode)
            writer.add_scalar("Evaluation/Lethal", mean_lethal, episode)
            writer.add_scalar(
                "Evaluation/Draw first w/ Backflip", mean_draw_first_w_backflip, episode
            )
            writer.add_scalar(
                "Evaluation/Use Dagger Throw vs. Strike", mean_dagger_throw_vs_strike, episode
            )

        # Save model
        if (episode % save_every) == 0:
            torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")


# TODO: fix path
# TODO: parse values accordingly
def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    config["num_episodes"] = int(config["num_episodes"])
    config["coef_entropy_elbow"] = int(config["coef_entropy_elbow"])

    return config


# TODO: use **kwargs, improve signature
def _init_optimizer(name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
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
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = _load_config(config_path)

    # Models
    model = ActorCritic(config["model"]["dim_card"])

    # Optimizer
    optimizer = _init_optimizer(
        config["optimizer"]["name"], model, **config["optimizer"]["kwargs"]
    )

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
        config["eval_every"],
        config["save_every"],
        model,
        optimizer,
        torch.device("cpu"),
        config["gamma"],
        coefs_entropy,
        config["max_grad_norm"],
        config["num_eval"],
    )
