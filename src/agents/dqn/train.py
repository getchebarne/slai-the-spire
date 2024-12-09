import os
import random
import shutil

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
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat


# TODO: add discount to config
def _train_on_batch(
    batch: Batch,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    discount: float = 0.99,
) -> float:
    # Set model to train mode
    model.train()

    # Forward pass
    q_ts = model(batch.state_ts.to(device)).gather(1, batch.actions.view((-1, 1)).to(device))

    # Predict next state's Q values
    with torch.no_grad():
        q_tp1s = model(batch.state_tp1s.to(device))

    # Get maximum next state's Q value
    q_tp1_maxs, _ = torch.max(q_tp1s - (1 - batch.valid_action_mask_tp1s.to(device)) * 1e20, dim=1)

    # Calculate target
    q_targets = (
        batch.rewards.to(device) + discount * q_tp1_maxs * (1 - batch.game_over_flags.to(device))
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
    combat_view: CombatView,
    combat_view_encoded: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    epsilon: float,
) -> int:
    valid_action_mask = get_valid_action_mask(combat_view)

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
            - (1 - torch.tensor(valid_action_mask, dtype=torch.long)) * 1e20
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
    model: DQNAgent,
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

        # Get combat view
        combat_view_t = view_combat(combat_manager)
        combat_view_t_encoded = encode_combat_view(combat_view_t, torch.device("cpu"))

        # Get action from agent
        action_idx = _get_action_idx(combat_view_t, combat_view_t_encoded, model, device, epsilon)

        # Game step
        action = action_idx_to_action(action_idx, combat_view_t)
        step(combat_manager, action)

        # Get new entities, new valid actions, game over flag and instant reward
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
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            continue

        loss_batch = _train_on_batch(batch, model, optimizer, device)
        loss_episode += loss_batch

    return combat_manager, loss_episode / num_moves, num_moves


def train(
    exp_name: str,
    num_episodes: int,
    buffer_size: int,
    batch_size: int,
    eval_every: int,
    save_every: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    value_start: float,
    value_end: float,
    episode_elbow: int,
    device: torch.device,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Instantiate agent
    agent = DQNAgent(model)

    # Explorer
    epsilons = linear_decay(num_episodes, episode_elbow, value_start, value_end)

    # Send model to device TODO move
    model = model.to(device)

    # Train
    for epoch in range(num_episodes):
        combat_manager, epoch_loss, num_moves = _play_episode(
            model, optimizer, replay_buffer, epsilons[epoch], batch_size, device
        )
        # View end state
        combat_view_end = view_combat(combat_manager)

        writer.add_scalar("Epoch loss", epoch_loss, epoch)
        writer.add_scalar("Number of moves", num_moves, epoch)
        writer.add_scalar("Epsilon", epsilons[epoch], epoch)
        writer.add_scalar("Final HP", combat_view_end.character.health.current, epoch)

        # Evaluate
        if (epoch % eval_every) == 0:
            final_hp = _evaluate_agent(agent)
            writer.add_scalar("Eval/Final HP", final_hp, epoch)

        # Save model
        if (epoch % save_every) == 0:
            torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")


# TODO: fix path
# TODO: parse values accordingly
def _load_config(config_path: str = "src/agents/dqn/config.yml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# TODO: use **kwargs, improve signature
def _init_model(config_model: dict) -> nn.Module:
    return getattr(models, config_model["name"])(**config_model["kwargs"])


# TODO: use **kwargs, improve signature
def _init_optimizer(config_optimizer: dict, model: nn.Module) -> torch.optim.Optimizer:
    return getattr(torch.optim, config_optimizer["name"])(
        **config_optimizer["kwargs"], params=model.parameters()
    )


if __name__ == "__main__":
    config = _load_config()
    model = _init_model(config["model"])
    optimizer = _init_optimizer(config["optimizer"], model)

    # Copy config
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy("src/agents/dqn/config.yml", f"experiments/{config['exp_name']}/config.yml")

    train(
        config["exp_name"],
        int(config["num_episodes"]),
        int(config["buffer_size"]),
        config["batch_size"],
        config["eval_every"],
        config["save_every"],
        model,
        optimizer,
        config["value_start"],
        config["value_end"],
        config["episode_elbow"],
        torch.device("cpu"),
    )
