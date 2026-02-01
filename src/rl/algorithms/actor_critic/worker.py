"""
Worker process for parallel environment stepping.

Each worker maintains its own game state and receives actions from the master.
"""

from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection

from src.game.action import Action
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.constants import ASCENSION_LEVEL
from src.rl.reward import compute_reward


class Command(Enum):
    STEP = "STEP"
    RESET = "RESET"
    CLOSE = "CLOSE"


@dataclass
class WorkerData:
    """Data sent from worker to master after each step/reset."""

    view_game_state: ViewGameState
    fsm: FSM
    game_over: bool
    reward: float | None = None


def worker(conn: Connection) -> None:
    """
    Worker process that maintains a game environment.

    Receives commands from master:
    - RESET: Create new game, return initial state
    - STEP: Execute action, return new state and reward
    - CLOSE: Terminate worker
    """
    game_state = None
    view_game_state = None

    while True:
        command, payload = conn.recv()

        if command == Command.CLOSE:
            break

        elif command == Command.RESET:
            # Create new game
            game_state = create_game_state(ASCENSION_LEVEL)
            initialize_game_state(game_state)

            # Get initial view
            view_game_state = get_view_game_state(game_state)
            game_over = game_state.fsm == FSM.GAME_OVER

            if game_over:
                raise RuntimeError("Game started in game over state")

            conn.send(WorkerData(
                view_game_state=view_game_state,
                fsm=game_state.fsm,
                game_over=game_over,
            ))

        elif command == Command.STEP:
            if game_state is None:
                raise RuntimeError("Must reset before stepping")

            if game_state.fsm == FSM.GAME_OVER:
                raise RuntimeError("Cannot step in game over state")

            # Payload is the Action to execute
            action: Action = payload

            # Execute action
            step(game_state, action, fast_mode=True)

            # Get new state
            view_game_state_next = get_view_game_state(game_state)
            game_over = game_state.fsm == FSM.GAME_OVER

            # Compute reward
            reward = compute_reward(view_game_state, view_game_state_next, game_over)

            # Update current view
            view_game_state = view_game_state_next

            conn.send(WorkerData(
                view_game_state=view_game_state,
                fsm=game_state.fsm,
                game_over=game_over,
                reward=reward,
            ))
