import os
import random
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn.encode import encode_combat_view
from src.agents.dqn.explorer import linear_decay
from src.agents.dqn.memory import Batch
from src.agents.dqn.memory import ReplayBuffer
from src.agents.dqn.memory import Sample
from src.agents.dqn.model import DeepQNetwork
from src.agents.dqn.model import action_idx_to_action
from src.agents.dqn.model import get_valid_action_mask
from src.agents.evaluation import evaluate_blunder
from src.agents.evaluation import evaluate_lethal
from src.agents.reward import compute_reward
from src.game.combat.create import create_combat_state
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.state import CombatState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# TODO: parametrize
TOLERANCE = 100


# TODO: add discount to config
def _train_on_batch(
    batch: Batch,
    model_online: nn.Module,
    model_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    discount: float,
) -> float:
    # Forward pass
    q_values_all = model_online(batch.states.to(device))
    q_values_taken = q_values_all.gather(1, batch.actions.view(-1, 1).to(device))

    # Predict next state's Q values with both the online and target models
    states_next_device = batch.states_next.to(device)
    with torch.no_grad():
        q_values_next_online = model_online(states_next_device)
        q_values_next_target = model_target(states_next_device)

    # Select the action with the highest Q value using the online model
    q_values_next_online[~batch.valid_action_masks_next.to(device)] = float("-inf")
    best_action_next_idxs = torch.argmax(q_values_next_online, dim=1, keepdim=True)

    # Get the Q value of the selected best action from the target model
    q_values_next_target_max = q_values_next_target.gather(1, best_action_next_idxs)

    # Calculate TD target
    q_targets = (
        batch.rewards.view(-1, 1).to(device)
        + discount * q_values_next_target_max * (1 - batch.game_over_flags.view(-1, 1).to(device))
    ).view(-1, 1)

    # Backward pass
    optimizer.zero_grad()
    loss = F.mse_loss(q_values_taken, q_targets)
    loss.backward()

    # Apply gradients
    optimizer.step()

    # Return batch loss
    return loss.item()


def _evaluate_agent(model: DeepQNetwork, device: torch.device) -> int:
    # Get new game
    cs = create_combat_state()
    start_combat(cs)

    while not is_game_over(cs.entity_manager):
        combat_view = view_combat(cs)
        combat_view_tensor = encode_combat_view(combat_view, device)
        valid_action_mask_tensor = torch.tensor(
            [get_valid_action_mask(combat_view)], dtype=torch.bool, device=device
        )

        # Get action
        with torch.no_grad():
            q_values = model(combat_view_tensor.unsqueeze(0))

        q_values[~valid_action_mask_tensor] = float("-inf")
        action_idx = torch.argmax(q_values).item()
        action = action_idx_to_action(action_idx, combat_view)

        # Game step
        step(cs, action)

    # Return final health
    return view_combat(cs).character.health_current


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
    discount: float,
) -> tuple[CombatState, float, int]:
    # Get new game
    cs = create_combat_state()
    start_combat(cs)

    # Intialize variables
    combat_view = view_combat(cs)
    combat_view_tensor = encode_combat_view(combat_view, device)
    valid_action_mask = get_valid_action_mask(combat_view)
    valid_action_mask_tensor = torch.tensor([valid_action_mask], dtype=torch.bool, device=device)
    game_over_flag = False
    loss_episode = 0
    num_moves = 0

    # Start playing
    while not game_over_flag:
        num_moves += 1

        # Get action from agent
        if random.uniform(0, 1) < epsilon:
            # Explore
            action_idx = random.choice(
                [action_idx for action_idx, is_valid in enumerate(valid_action_mask) if is_valid]
            )
        else:
            # Exploit
            with torch.no_grad():
                q_values = model_online(combat_view_tensor.unsqueeze(0))

            q_values[~valid_action_mask_tensor] = float("-inf")
            action_idx = torch.argmax(q_values).item()

        # Game step
        action = action_idx_to_action(action_idx, combat_view)
        step(cs, action)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_next = view_combat(cs)
        combat_view_tensor_next = encode_combat_view(combat_view_next, device)
        valid_action_mask_next = get_valid_action_mask(combat_view_next)
        valid_action_mask_tensor_next = torch.tensor(
            [valid_action_mask_next], dtype=torch.bool, device=device
        )
        game_over_flag_next = is_game_over(cs.entity_manager)
        reward = compute_reward(combat_view, combat_view_next, game_over_flag_next)

        # Store transition in memory
        replay_buffer.store(
            Sample(
                combat_view_tensor,
                combat_view_tensor_next,
                action_idx,
                reward,
                valid_action_mask_tensor_next,
                game_over_flag_next,
            )
        )

        # Update state, valid actions, and game over flag
        combat_view = combat_view_next
        combat_view_tensor = combat_view_tensor_next
        valid_action_mask = valid_action_mask_next
        valid_action_mask_tensor = valid_action_mask_tensor_next
        game_over_flag = game_over_flag_next

        # Check if there's enough sample to start fitting the network
        if replay_buffer.num_samples < batch_size * TOLERANCE:
            continue

        # Fit
        batch = replay_buffer.sample(batch_size)
        loss_batch = _train_on_batch(
            batch, model_online, model_target, optimizer, device, discount
        )

        # Accumulate episode loss
        loss_episode += loss_batch

        # Increase global step counter and transfer parameters if needed
        step_global += 1
        if (step_global % transfer_every) == 0:
            _transfer_params(model_online, model_target)

    return cs, loss_episode / num_moves, step_global


def train(
    exp_name: str,
    num_episodes: int,
    buffer_size: int,
    batch_size: int,
    log_every: int,
    save_every: int,
    transfer_every: int,
    model_online: nn.Module,
    model_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilons: list[float],
    device: torch.device,
    discount: float,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Send models to device TODO: move
    model_online.to(device)
    model_target.to(device)

    # Train
    step_global = 0
    for episode in range(num_episodes):
        cs, loss, step_global = _play_episode(
            step_global,
            model_online,
            model_target,
            transfer_every,
            optimizer,
            replay_buffer,
            epsilons[episode],
            batch_size,
            device,
            discount,
        )
        # View end state
        combat_view_end = view_combat(cs)

        if (episode % log_every) == 0:
            writer.add_scalar("Loss", loss, episode)
            writer.add_scalar("Epsilon", epsilons[episode], episode)
            writer.add_scalar("Final HP", combat_view_end.character.health_current, episode)

            # TODO: parametrize this
            num_eval = 10
            hp_final = []
            blunder = []
            lethal = []
            for _ in range(num_eval):
                hp_final.append(_evaluate_agent(model_online, device))
                blunder.append(evaluate_blunder(model_online, device))
                lethal.append(evaluate_lethal(model_online, device))

            writer.add_scalar("Evaluation/Health", sum(hp_final) / num_eval, episode)
            writer.add_scalar("Scenario/Blunder", sum(blunder) / num_eval, episode)
            writer.add_scalar("Scenario/Lethal", sum(lethal) / num_eval, episode)

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
def _init_optimizer(name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, name)(**kwargs, params=model.parameters())


def _transfer_params(model_online: nn.Module, model_target: nn.Module) -> None:
    for param_online, param_target in zip(model_online.parameters(), model_target.parameters()):
        param_target.data.copy_(param_online.data)


if __name__ == "__main__":
    config = _load_config()
    model_online = DeepQNetwork(config["model"]["dim_card"])
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
        config["save_every"],
        config["transfer_every"],
        model_online,
        model_target,
        optimizer,
        epsilons,
        torch.device("cpu"),
        config["discount"],
    )
