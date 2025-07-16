import os
import random
import shutil
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.game.combat.create import create_game_state
from src.game.combat.view import view_combat
from src.game.main import start_combat
from src.game.main import step
from src.game.utils import is_game_over
from src.rl.algorithms.dqn.explorer import linear_decay
from src.rl.algorithms.dqn.memory import Batch
from src.rl.algorithms.dqn.memory import ReplayBuffer
from src.rl.encoding import encode_combat_view
from src.rl.encoding import pack_combat_view_encoding
from src.rl.encoding import unpack_combat_view_encoding
from src.rl.evaluation import evaluate_blunder
from src.rl.evaluation import evaluate_dagger_throw_vs_strike
from src.rl.evaluation import evaluate_draw_first_w_backflip
from src.rl.evaluation import evaluate_final_hp
from src.rl.evaluation import evaluate_lethal
from src.rl.models.dqn import DeepQNetwork
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask
from src.rl.policies import PolicyQMax
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# TODO: add discount to config
def _train_on_batch(
    batch: Batch,
    model_online: DeepQNetwork,
    model_target: DeepQNetwork,
    optimizer: torch.optim.Optimizer,
    discount: float,
) -> float:
    # Forward pass
    q_values_all = model_online(*unpack_combat_view_encoding(batch.states).as_tuple())
    q_values_taken = q_values_all.gather(1, batch.actions)

    # Predict next state's Q values with both the online and target models
    with torch.no_grad():
        q_values_next_online = model_online(
            *unpack_combat_view_encoding(batch.states_next).as_tuple()
        )
        q_values_next_target = model_target(
            *unpack_combat_view_encoding(batch.states_next).as_tuple()
        )

    # Select the action with the highest Q value using the online model
    q_values_next_online[~batch.valid_action_masks_next] = float("-inf")
    best_action_next_idxs = torch.argmax(q_values_next_online, dim=1, keepdim=True)

    # Get the Q value of the selected best action from the target model
    q_values_next_target_max = q_values_next_target.gather(1, best_action_next_idxs)

    # Calculate TD target
    q_targets = batch.rewards + discount * q_values_next_target_max * (1 - batch.game_over_flags)

    # Backward pass
    optimizer.zero_grad()
    loss = F.mse_loss(q_values_taken, q_targets)
    loss.backward()

    # Apply gradients
    optimizer.step()

    # Return batch loss
    return loss


def _play_episode(
    step_global: int,
    model_online: DeepQNetwork,
    model_target: DeepQNetwork,
    transfer_every: int,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    batch_size: int,
    min_samples_to_start_training: int,
    discount: float,
    device: torch.device,
) -> tuple[float | None, int]:
    # Get new game
    cs = create_game_state()
    start_combat(cs)

    # Intialize variables
    combat_view = view_combat(cs)
    combat_view_encoding = encode_combat_view(combat_view, device)
    valid_action_mask = get_valid_action_mask(combat_view)
    valid_action_mask_tensor = torch.tensor([valid_action_mask], dtype=torch.bool, device=device)
    game_over_flag = False
    loss_episode = None
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
                q_values = model_online(*combat_view_encoding.as_tuple())

            q_values[~valid_action_mask_tensor] = float("-inf")
            action_idx = torch.argmax(q_values, dim=1).item()

        # Game step
        action = action_idx_to_action(action_idx, combat_view)
        step(cs, action)

        # Get new state, new valid actions, game over flag and instant reward
        combat_view_next = view_combat(cs)
        combat_view_encoding_next = encode_combat_view(combat_view_next, device)
        valid_action_mask_next = get_valid_action_mask(combat_view_next)
        valid_action_mask_tensor_next = torch.tensor(
            [valid_action_mask_next], dtype=torch.bool, device=device
        )
        game_over_flag_next = is_game_over(cs.entity_manager)
        reward = compute_reward(combat_view, combat_view_next, game_over_flag_next)

        # Store transition in memory
        replay_buffer.store(
            pack_combat_view_encoding(combat_view_encoding).squeeze(),
            pack_combat_view_encoding(combat_view_encoding_next).squeeze(),
            action_idx,
            reward,
            valid_action_mask_tensor_next,
            game_over_flag_next,
        )

        # Update state, valid actions, and game over flag
        combat_view = combat_view_next
        combat_view_encoding = combat_view_encoding_next
        valid_action_mask = valid_action_mask_next
        valid_action_mask_tensor = valid_action_mask_tensor_next
        game_over_flag = game_over_flag_next

        # Check if there's enough samples to start fitting the network
        if replay_buffer.num_samples < min_samples_to_start_training:
            continue

        # Fit
        batch = replay_buffer.sample(batch_size)
        loss_batch = _train_on_batch(batch, model_online, model_target, optimizer, discount)

        # Accumulate episode loss
        if loss_episode is None:
            loss_episode = torch.tensor(0.0, dtype=torch.float32, device=device)

        loss_episode += loss_batch

        # Increase global step counter and transfer parameters if needed
        step_global += 1
        if (step_global % transfer_every) == 0:
            _transfer_params(model_online, model_target)

    # Episode end
    loss_episode_mean = None if loss_episode is None else loss_episode.item() / num_moves

    return loss_episode_mean, step_global


def train(
    exp_name: str,
    num_episodes: int,
    buffer_size: int,
    batch_size: int,
    log_every: int,
    save_every: int,
    transfer_every: int,
    model_online: DeepQNetwork,
    model_target: DeepQNetwork,
    optimizer: torch.optim.Optimizer,
    epsilons: list[float],
    num_eval: int,
    min_samples_to_start_training: int,
    discount: float,
    device: torch.device,
) -> None:
    # Initialize objects
    writer = SummaryWriter(f"experiments/{exp_name}")
    replay_buffer = ReplayBuffer(buffer_size, device)

    # Send models to device
    model_online.to(device)
    model_target.to(device)

    # Initialize policy
    policy = PolicyQMax(model_online, device)

    # Train
    step_global = 0
    for episode in range(num_episodes):
        epsilon = epsilons[episode]
        loss, step_global = _play_episode(
            step_global,
            model_online,
            model_target,
            transfer_every,
            optimizer,
            replay_buffer,
            epsilon,
            batch_size,
            min_samples_to_start_training,
            discount,
            device,
        )
        if (episode % log_every) == 0:
            if loss is not None:
                writer.add_scalar("Loss", loss, episode)

            writer.add_scalar("Epsilon", epsilon, episode)

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
            torch.save(model_online.state_dict(), f"experiments/{exp_name}/model.pth")


def _transfer_params(model_online: DeepQNetwork, model_target: DeepQNetwork) -> None:
    for param_online, param_target in zip(model_online.parameters(), model_target.parameters()):
        param_target.data.copy_(param_online.data)


if __name__ == "__main__":
    config_path = "src/rl/algorithms/dqn/config.yml"
    config = load_config(config_path)
    model_online = DeepQNetwork(**config["model"])
    model_target = deepcopy(model_online)
    optimizer = init_optimizer(
        config["optimizer"]["name"], model_online, **config["optimizer"]["kwargs"]
    )

    # Copy config, and other important scripts to replicate results TODO: improve
    os.makedirs(f"experiments/{config['exp_name']}")
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")
    shutil.copy("src/rl/encoding.py", f"experiments/{config['exp_name']}/encoding.py")
    shutil.copy("src/rl/reward.py", f"experiments/{config['exp_name']}/reward.py")
    shutil.copy("src/rl/models/dqn.py", f"experiments/{config['exp_name']}/dqn.py")

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
        config["num_eval"],
        config["min_samples_to_start_training"],
        config["discount"],
        torch.device("cpu"),
    )
