from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection

import torch

from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.state import get_view_game_state
from src.rl.encoding.state import encode_view_game_state
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask
from src.rl.reward import compute_reward


_ASCENSION_LEVEL = 1


class Command(Enum):
    STEP = "STEP"
    RESET = "RESET"


@dataclass
class WorkerData:
    encoding_game_state: tuple[torch.Tensor, ...]
    valid_action_mask: torch.Tensor
    game_over_flag: bool
    reward: float | None = None


def worker(remote: Connection, device: torch.device) -> None:
    # Intialize variables to `None`
    game_state = None
    game_over_flag = None
    view_game_state = None

    while True:
        # Receive data from the master
        command, action_idx = remote.recv()

        if command == Command.RESET:
            # Reset combat
            game_state = create_game_state(_ASCENSION_LEVEL)
            initialize_game_state(game_state)

            # Get combat view, encode it, valid action mask, and game over flag
            view_game_state = get_view_game_state(game_state)
            encoding_game_state = encode_view_game_state(view_game_state, device)
            valid_action_mask = get_valid_action_mask(view_game_state)
            game_over_flag = game_state.fsm == FSM.GAME_OVER

            if game_over_flag:
                raise RuntimeError("Game started in game over state")

            remote.send(
                WorkerData(
                    encoding_game_state,
                    valid_action_mask.view(1, -1),
                    game_over_flag,
                )
            )

        if command == Command.STEP:
            # If the game's over, raise an error
            if game_over_flag:
                raise RuntimeError("Attempted to make an action while the game's over.")

            # Game step
            action = action_idx_to_action(action_idx)
            step(game_state, action)

            # Get next combat view, compute game over flag and reward
            view_game_state_next = get_view_game_state(game_state)
            game_over_flag = game_state.fsm == FSM.GAME_OVER
            reward = compute_reward(view_game_state, view_game_state_next, game_over_flag)

            # Update current state and valid action mask
            view_game_state = view_game_state_next
            valid_action_mask = get_valid_action_mask(view_game_state)

            # Send transition data
            remote.send(
                WorkerData(
                    encode_view_game_state(view_game_state, device),
                    valid_action_mask.view(1, -1),
                    game_over_flag,
                    reward,
                )
            )
