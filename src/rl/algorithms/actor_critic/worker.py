from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection

import torch

from src.game.combat.create import create_game_state
from src.game.combat.view import view_combat
from src.game.main import start_combat
from src.game.main import step
from src.game.utils import is_game_over
from src.rl.encoding import encode_combat_view
from src.rl.encoding import pack_combat_view_encoding
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask
from src.rl.reward import compute_reward


class Command(Enum):
    STEP = "STEP"
    RESET = "RESET"


@dataclass
class WorkerData:
    combat_view_encoding: torch.Tensor
    valid_action_mask: list[bool]
    game_over_flag: bool
    reward: float | None = None


def worker(remote: Connection, device: torch.device) -> None:
    # Intialize variables to `None`
    cs = None
    game_over_flag = None
    combat_view = None

    while True:
        # Receive data from the master
        command, action_idx = remote.recv()

        if command == Command.RESET:
            # Reset combat
            cs = create_game_state()
            start_combat(cs)

            # Get combat view, encode it, valid action mask, and game over flag
            combat_view = view_combat(cs)
            combat_view_encoding = encode_combat_view(combat_view, device)
            valid_action_mask = get_valid_action_mask(combat_view)
            game_over_flag = is_game_over(cs.entity_manager)

            if game_over_flag:
                raise RuntimeError("Game started in game over state")

            remote.send(
                WorkerData(
                    pack_combat_view_encoding(combat_view_encoding),
                    valid_action_mask,
                    game_over_flag,
                )
            )

        if command == Command.STEP:
            # If the game's over, raise an error
            if game_over_flag:
                raise RuntimeError("Attempted to take an action while the game's over.")

            # Game step
            action = action_idx_to_action(action_idx, combat_view)
            step(cs, action)

            # Get next combat view, compute game over flag and reward
            combat_view_next = view_combat(cs)
            game_over_flag = is_game_over(cs.entity_manager)
            reward = compute_reward(combat_view, combat_view_next, game_over_flag)

            # Update current state and valid action mask
            combat_view = combat_view_next
            valid_action_mask = get_valid_action_mask(combat_view)

            # Send transition data
            remote.send(
                WorkerData(
                    pack_combat_view_encoding(encode_combat_view(combat_view, device)),
                    valid_action_mask,
                    game_over_flag,
                    reward,
                )
            )
