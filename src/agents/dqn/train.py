import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn.explorer import linear_decay
from src.agents.dqn.memory import Batch
from src.agents.dqn.memory import ReplayBuffer
from src.agents.dqn.memory import Sample
from src.agents.dqn.model import DQNAgent
from src.agents.dqn.utils import get_valid_action_mask
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import combat_start
from src.game.combat.phase import turn_monster
from src.game.combat.state import GameState
from src.game.combat.utils import is_game_over
from src.game.combat.utils import new_game
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat


def _action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 5:
        return Action(ActionType.SELECT_ENTITY, combat_view.hand[action_idx].entity_id)

    if action_idx == 5:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[5 - action_idx].entity_id)

    if action_idx == 6:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


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


def _play_episode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
) -> tuple[GameState, float]:
    # Get new game
    state = new_game()
    combat_start(state)

    # Start playing
    loss_episode = 0
    num_moves = 0
    while not is_game_over(state):
        if not state.actor_turn_id == state.character_id:
            turn_monster(state)
            continue

        num_moves += 1

        # Get combat view
        combat_view_t = view_combat(state)
        valid_action_mask_t = get_valid_action_mask(combat_view_t)

        if random.uniform(0, 1) < epsilon:
            # Explore
            action_idx = random.choice(
                [i for i, is_valid in enumerate(valid_action_mask_t) if is_valid]
            )

        else:
            # Exploit
            with torch.no_grad():
                q_t = model([combat_view_t])

            action_idx = (
                (q_t - (1 - torch.tensor(valid_action_mask_t, dtype=torch.int)) * 1e20)
                .argmax()
                .item()
            )

        # Get action from action index
        action = _action_idx_to_action(action_idx, combat_view_t)

        # Update game's state
        handle_action(state, action)
        turn_monster(state)

        # Get new state, new valid actions and game over flag
        combat_view_tp1 = view_combat(state)
        valid_action_mask_tp1 = get_valid_action_mask(combat_view_tp1)
        game_over_flag = is_game_over(state)

        # Compute instant reward
        if game_over_flag:
            if combat_view_tp1.character.health.current > 0:
                reward = (
                    combat_view_tp1.character.health.current / combat_view_tp1.character.health.max
                )
            else:
                reward = -1

        else:
            reward = 0

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

        loss_batch = _train_on_batch(batch, model, optimizer)
        loss_episode += loss_batch

    return state, loss_episode / num_moves


def train() -> None:
    # Config
    buffer_size = int(5e3)
    num_epochs = int(1e3)
    batch_size = 48
    writer = SummaryWriter("experiments/test-del")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Get model
    model = DQNAgent(4)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Explorer
    epsilons = linear_decay(num_epochs, num_epochs // 2, 1, 0.001)

    # Train
    for epoch in range(num_epochs):
        state, epoch_loss = _play_episode(
            model, optimizer, replay_buffer, epsilons[epoch], batch_size
        )
        # View end state
        combat_view_end = view_combat(state)

        writer.add_scalar("Epoch loss", epoch_loss, epoch)
        writer.add_scalar("Epsilon", epsilons[epoch], epoch)
        writer.add_scalar("Final HP", combat_view_end.character.health.current, epoch)
        # writer.add_scalar("Blunders", blunders, epoch)


if __name__ == "__main__":
    train()
