import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn.explorer import linear_decay
from src.agents.dqn.memory import Batch
from src.agents.dqn.memory import ReplayBuffer
from src.agents.dqn.memory import Sample
from src.agents.dqn.model import EmbeddingMLP
from src.agents.dqn.reward import compute_reward
from src.agents.dqn.utils import action_idx_to_action
from src.agents.dqn.utils import get_valid_action_mask
from src.agents.dqn_a import DQNAgent
from src.game.combat.view import CombatView
from src.game.combat.create import create_combat
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import turn_monster
from src.game.combat.state import GameState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def _train_on_batch(
    batch: Batch, model: nn.Module, optimizer: torch.optim.Optimizer, discount: float = 0.99
) -> float:
    # Set model to train model
    model.train()

    # Forward pass
    q_ts = model(batch.state_ts).gather(
        1,
        torch.tensor(batch.actions, dtype=torch.int64).reshape(-1, 1),
    )
    # Predict next state's Q values
    with torch.no_grad():
        q_tp1s = model(batch.state_tp1s)

    # Get maximum next state's Q value
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


def _game_step(state: GameState, agent: DQNAgent, epsilon: float) -> tuple[CombatView, int]:
    # Get combat view
    combat_view = view_combat(state)
    valid_action_mask = get_valid_action_mask(combat_view)

    if random.uniform(0, 1) < epsilon:
        # Explore
        action_idx = random.choice([i for i, is_valid in enumerate(valid_action_mask) if is_valid])

    else:
        # Exploit
        action_idx = agent.select_action(combat_view, valid_action_mask)

    # Handle action
    action = action_idx_to_action(action_idx, combat_view)
    handle_action(state, action)

    # Monsters' turn
    turn_monster(state)

    return combat_view, action_idx


def _play_episode(
    agent: DQNAgent,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
) -> tuple[GameState, float]:
    # Get new game
    state = create_combat()

    # Start playing
    loss_episode = 0
    num_moves = 0
    while not is_game_over(state):
        num_moves += 1

        combat_view_t, action_idx = _game_step(state, agent, epsilon)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_tp1 = view_combat(state)
        valid_action_mask_tp1 = get_valid_action_mask(combat_view_tp1)
        game_over_flag = is_game_over(state)
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

    return state, loss_episode / num_moves


def train() -> None:
    # Config
    buffer_size = int(1e3)
    num_epochs = int(5e2)
    batch_size = 48
    writer = SummaryWriter("experiments/test-del-9")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Get model
    model = EmbeddingMLP(4)
    agent = DQNAgent(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Explorer
    epsilons = linear_decay(num_epochs, num_epochs // 2, 1, 0.001)

    # Train
    for epoch in range(num_epochs):
        state, epoch_loss = _play_episode(
            agent, optimizer, replay_buffer, epsilons[epoch], batch_size
        )
        # View end state
        combat_view_end = view_combat(state)

        writer.add_scalar("Epoch loss", epoch_loss, epoch)
        writer.add_scalar("Epsilon", epsilons[epoch], epoch)
        writer.add_scalar("Final HP", combat_view_end.character.health.current, epoch)
        # writer.add_scalar("Blunders", blunders, epoch)


if __name__ == "__main__":
    train()
