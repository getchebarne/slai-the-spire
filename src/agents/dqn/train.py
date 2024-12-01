import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn import models
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


def _train_on_batch(
    batch: Batch, model: nn.Module, optimizer: torch.optim.Optimizer, discount: float = 0.99
) -> float:
    # Set model to train mode
    model.train()

    # Forward pass
    q_ts = model(batch.entities_ts).gather(
        1,
        torch.tensor(batch.actions, dtype=torch.int64).reshape(-1, 1),
    )
    # Predict next entities's Q values
    with torch.no_grad():
        q_tp1s = model(batch.entities_tp1s)

    # Get maximum next entities's Q value
    q_tp1_maxs, _ = torch.max(
        q_tp1s - (1 - torch.tensor(batch.valid_action_mask_tp1s, dtype=torch.int)) * 1e20, dim=1
    )

    # Calculate target
    q_targets = (
        torch.tensor(batch.rewards)
        + discount * q_tp1_maxs * (1 - torch.tensor(batch.game_over_flags, dtype=torch.int))
    ).reshape(-1, 1)

    # Backward pass
    optimizer.zero_grad()
    loss = F.mse_loss(q_ts, q_targets)
    loss.backward()

    # Apply gradients
    optimizer.step()

    # Return batch loss
    return loss.item()


def _get_action_idx(combat_view: CombatView, agent: DQNAgent, epsilon: float) -> int:
    valid_action_mask = get_valid_action_mask(combat_view)  # TODO: FIX

    if random.uniform(0, 1) < epsilon:
        # Explore
        return random.choice([i for i, is_valid in enumerate(valid_action_mask) if is_valid])

    # Exploit
    return agent.select_action(combat_view, valid_action_mask)


def _evaluate_agent(agent: DQNAgent) -> int:
    # Get new game TODO: improve this
    combat_manager = create_combat_manager()
    combat_start(combat_manager)
    process(combat_manager)

    while not is_game_over(combat_manager.entities):
        # Get combat view
        combat_view_t = view_combat(combat_manager)

        # Get action from agent
        action_idx = _get_action_idx(combat_view_t, agent, epsilon=-1)
        action = action_idx_to_action(action_idx, combat_view_t)

        # Game step
        step(combat_manager, action)

    # View end entities
    combat_view_end = view_combat(combat_manager)

    return combat_view_end.character.health.current


def _play_episode(
    agent: DQNAgent,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
) -> tuple[CombatManager, float]:
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

        # Get action from agent
        action_idx = _get_action_idx(combat_view_t, agent, epsilon)
        action = action_idx_to_action(action_idx, combat_view_t)

        # Game step
        step(combat_manager, action)

        # Get new entities, new valid actions, game over flag and instant reward
        combat_view_tp1 = view_combat(combat_manager)
        valid_action_mask_tp1 = get_valid_action_mask(combat_view_tp1)
        game_over_flag = is_game_over(combat_manager.entities)
        reward = compute_reward(combat_view_tp1, game_over_flag)

        # Store transition in memory
        replay_buffer.store(
            Sample(
                combat_view_t,
                combat_view_tp1,
                action_idx,
                reward,
                valid_action_mask_tp1,
                game_over_flag,
            )
        )

        # Train
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            continue

        loss_batch = _train_on_batch(batch, agent.model, optimizer)
        loss_episode += loss_batch

    return combat_manager, loss_episode / num_moves


def train(
    exp_name: str,
    num_episodes: int,
    buffer_size: int,
    batch_size: int,
    eval_every: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    value_start: float,
    value_end: float,
    episode_elbow: int,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Instantiate agent
    agent = DQNAgent(model)

    # Explorer
    epsilons = linear_decay(num_episodes, episode_elbow, value_start, value_end)

    # Train
    for epoch in range(num_episodes):
        combat_manager, epoch_loss = _play_episode(
            agent, optimizer, replay_buffer, epsilons[epoch], batch_size
        )
        # View end state
        combat_view_end = view_combat(combat_manager)

        writer.add_scalar("Epoch loss", epoch_loss, epoch)
        writer.add_scalar("Epsilon", epsilons[epoch], epoch)
        writer.add_scalar("Final HP", combat_view_end.character.health.current, epoch)

        if (epoch % eval_every) == 0:
            final_hp = _evaluate_agent(agent)
            writer.add_scalar("Eval/Final HP", final_hp, epoch)


# TODO: fix path
# TODO: parse values accordingly
def _load_config(config_path: str = "src/agents/dqn/config.yml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _init_model(config_model: dict) -> nn.Module:
    return getattr(models, config_model["name"])(**config_model["kwargs"])


def _init_optimizer(config_optimizer: dict, model: nn.Module) -> torch.optim.Optimizer:
    return getattr(torch.optim, config_optimizer["name"])(
        **config_optimizer["kwargs"], params=model.parameters()
    )


if __name__ == "__main__":
    config = _load_config()
    model = _init_model(config["model"])
    optimizer = _init_optimizer(config["optimizer"], model)

    train(
        config["exp_name"],
        int(config["num_episodes"]),
        int(config["buffer_size"]),
        config["batch_size"],
        config["eval_every"],
        model,
        optimizer,
        config["value_start"],
        config["value_end"],
        config["episode_elbow"],
    )
