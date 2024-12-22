import os
import random
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn import models
from src.agents.dqn.encoder import encode_combat_view
from src.agents.dqn.explorer import linear_decay
from src.agents.dqn.memory import Batch
from src.agents.dqn.memory import ReplayBuffer
from src.agents.dqn.memory import Sample
from src.agents.dqn.reward import compute_reward
from src.agents.dqn.utils import action_idx_to_action
from src.agents.dqn.utils import get_valid_action_mask
from src.agents.dqn_a import DQNAgent
from src.game.combat.create import create_combat_manager
from src.game.combat.main import process
from src.game.combat.main import step
from src.game.combat.manager import CombatManager
from src.game.combat.phase import combat_start
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# TODO: add discount to config
def _train_on_batch(
    batch: Batch,
    model_online: nn.Module,
    model_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    discount: float = 0.995,
) -> float:
    # Set model to train mode
    model_online.train()

    # Forward pass
    q_ts = model_online(batch.state_ts.to(device)).gather(1, batch.actions.view(-1, 1).to(device))

    # Predict next state's Q values with both the online and target models
    with torch.no_grad():
        state_tp1s_device = batch.state_tp1s.to(device)
        q_tp1s_online = model_online(state_tp1s_device)
        q_tp1s_target = model_target(state_tp1s_device)

        # Select the action with the highest Q value using the online model
        best_actions = torch.argmax(
            q_tp1s_online - 1e20 * (1 - batch.valid_action_mask_tp1s.to(device)),
            dim=1,
            keepdim=True,
        )

        # Get the Q value of the selected best action from the target model
        q_tp1_maxs = q_tp1s_target.gather(1, best_actions)

        # Calculate target
        q_targets = (
            batch.rewards.view(-1, 1).to(device)
            + discount * q_tp1_maxs * (1 - batch.game_over_flags.view(-1, 1).to(device))
        ).reshape(-1, 1)

    # Backward pass
    optimizer.zero_grad()
    loss = F.mse_loss(q_ts, q_targets)
    loss.backward()

    # Apply gradients
    optimizer.step()

    # Return batch loss
    return loss.item()


def _get_action_idx(
    valid_action_mask: list[bool],
    combat_view_encoded: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    epsilon: float,
) -> int:
    if random.uniform(0, 1) < epsilon:
        return random.choice(
            [action_idx for action_idx, is_valid in enumerate(valid_action_mask) if is_valid]
        )

    # Exploit
    with torch.no_grad():
        q_values = model(combat_view_encoded.unsqueeze(0).to(device))

    # Calculate action w/ highest q-value (masking invalid actions)
    action_idx = (
        (
            q_values.to(torch.device("cpu"))
            - (1 - torch.tensor(valid_action_mask, dtype=torch.float32)) * 1e20
        )
        .argmax()
        .item()
    )

    return action_idx


def _evaluate_agent(agent: DQNAgent) -> int:
    # Get new game TODO: improve this
    combat_manager = create_combat_manager()
    combat_start(combat_manager)
    process(combat_manager)

    while not is_game_over(combat_manager.entities):
        # Get combat view
        combat_view = view_combat(combat_manager)

        # Get action from agent
        action = agent.select_action(combat_view)

        # Game step
        step(combat_manager, action)

    # Return final health
    return view_combat(combat_manager).character.health.current


def _play_episode(
    step_global: int,
    model_online: nn.Module,
    model_target: nn.Module,
    transfer_every: int,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
    device: torch.device,
) -> tuple[CombatManager, float, int]:
    # Get new game TODO: improve this
    combat_manager = create_combat_manager()
    combat_start(combat_manager)
    process(combat_manager)

    # Start playing
    loss_episode = 0
    num_moves = 0
    while not is_game_over(combat_manager.entities):
        num_moves += 1

        # Get combat view, encode it, and get valid action mask
        combat_view_t = view_combat(combat_manager)
        combat_view_t_encoded = encode_combat_view(combat_view_t, torch.device("cpu"))
        valid_action_mask_t = get_valid_action_mask(combat_view_t)

        # Get action from agent
        action_idx = _get_action_idx(
            valid_action_mask_t, combat_view_t_encoded, model_online, device, epsilon
        )

        # Game step
        action = action_idx_to_action(action_idx, combat_view_t)
        step(combat_manager, action)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_tp1 = view_combat(combat_manager)
        valid_action_mask_tp1 = get_valid_action_mask(combat_view_tp1)
        game_over_flag = is_game_over(combat_manager.entities)
        reward = compute_reward(combat_view_t, combat_view_tp1, game_over_flag)

        # Store transition in memory
        replay_buffer.store(
            Sample(
                combat_view_t_encoded,
                encode_combat_view(combat_view_tp1, torch.device("cpu")),
                action_idx,
                reward,
                torch.tensor(valid_action_mask_tp1, dtype=torch.float32),
                game_over_flag,
            )
        )

        # Train
        if replay_buffer.num_samples < batch_size:
            continue

        # Sample buffer
        batch = replay_buffer.sample(batch_size)

        loss_batch = _train_on_batch(batch, model_online, model_target, optimizer, device)
        loss_episode += loss_batch
        step_global += 1
        if (step_global % transfer_every) == 0:
            _transfer_params(model_online, model_target)

    return combat_manager, loss_episode / num_moves, num_moves, step_global


def train(
    exp_name: str,
    num_episodes: int,
    buffer_size: int,
    batch_size: int,
    log_every: int,
    eval_every: int,
    save_every: int,
    transfer_every: int,
    model_online: nn.Module,
    model_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilons: list[float],
    device: torch.device,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Instantiate agent
    agent = DQNAgent(model_online)

    # Send models to device TODO: move
    model_online.to(device)
    model_target.to(device)

    # Train
    step_global = 0
    for episode in range(num_episodes):
        combat_manager, loss, num_moves, step_global = _play_episode(
            step_global,
            model_online,
            model_target,
            transfer_every,
            optimizer,
            replay_buffer,
            epsilons[episode],
            batch_size,
            device,
        )
        # View end state
        combat_view_end = view_combat(combat_manager)

        if (episode % log_every) == 0:
            writer.add_scalar("Loss", loss, episode)
            writer.add_scalar("Number of moves", num_moves, episode)
            writer.add_scalar("Epsilon", epsilons[episode], episode)
            writer.add_scalar("Final HP", combat_view_end.character.health.current, episode)

        # Evaluate
        if (episode % eval_every) == 0:
            hp_final = _evaluate_agent(agent)
            writer.add_scalar("Eval/Final HP", hp_final, episode)

        # Save model
        if (episode % save_every) == 0:
            torch.save(model_online.state_dict(), f"experiments/{exp_name}/model.pth")


# TODO: fix path
# TODO: parse values accordingly
def _load_config(config_path: str = "src/agents/dqn/config.yml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    config["num_episodes"] = int(config["num_episodes"])
    config["buffer_size"] = int(config["buffer_size"])

    return config


# TODO: use **kwargs, improve signature
def _init_model(name: str, **kwargs) -> nn.Module:
    return getattr(models, name)(**kwargs)


# TODO: use **kwargs, improve signature
def _init_optimizer(name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, name)(**kwargs, params=model.parameters())


def _transfer_params(model_online: nn.Module, model_target: nn.Module) -> None:
    for param_online, param_target in zip(model_online.parameters(), model_target.parameters()):
        param_target.data.copy_(param_online.data)


if __name__ == "__main__":
    config = _load_config()
    model_online = _init_model(config["model"]["name"], **config["model"]["kwargs"])
    model_target = deepcopy(model_online)
    optimizer = _init_optimizer(
        config["optimizer"]["name"], model_online, **config["optimizer"]["kwargs"]
    )

    # Copy config
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy("src/agents/dqn/config.yml", f"experiments/{config['exp_name']}/config.yml")

    # Get epsilons TODO add other epsilon schedules
    epsilons = linear_decay(
        config["num_episodes"], config["episode_elbow"], config["value_start"], config["value_end"]
    )

    train(
        config["exp_name"],
        config["num_episodes"],
        config["buffer_size"],
        config["batch_size"],
        config["log_every"],
        config["eval_every"],
        config["save_every"],
        config["transfer_every"],
        model_online,
        model_target,
        optimizer,
        epsilons,
        torch.device("cpu"),
    )
