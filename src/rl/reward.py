from src.game.view.state import ViewGameState


_PENALTY = -0.0050
_WEIGHT_HEALTH_CHAR = 0.0250
_WEIGHT_FLOOR = 0.1000


def compute_reward(
    view_game_state: ViewGameState, view_game_state_next: ViewGameState, game_over_flag: bool
) -> float:
    if game_over_flag:
        if view_game_state_next.character.health_current <= 0:
            # Loss
            return -1

        # Win
        return 1

    # Floor climb
    diff_floor = 0
    if (
        view_game_state.map.y_current is not None
        and view_game_state_next.map.y_current is not None
    ):
        diff_floor = view_game_state_next.map.y_current - view_game_state.map.y_current

    # Health difference
    diff_health_char = (
        view_game_state_next.character.health_current - view_game_state.character.health_current
    )

    return _WEIGHT_HEALTH_CHAR * diff_health_char + _WEIGHT_FLOOR * diff_floor + _PENALTY
