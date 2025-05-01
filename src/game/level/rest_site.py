from src.game.core.fsm import FSM
from src.game.state import GameState


def set_level_rest_site(game_state: GameState) -> None:
    # Clear effect queue
    game_state.effect_queue.clear()

    # Set FSM
    game_state.fsm = FSM.REST_SITE
