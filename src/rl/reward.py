from src.game.view.state import ViewGameState


_PENALTY = -0.0010
_WEIGHT_HEALTH_CHAR = 0.0250
_WEIGHT_FLOOR = 0.1000
_WEIGHT_UPGRADE = 0.5000  # Aprox. the reward you'd get from a full value rest


def compute_reward(
    view_game_state: ViewGameState, view_game_state_next: ViewGameState, game_over_flag: bool
) -> float:
    if game_over_flag:
        if view_game_state_next.character.health_current <= 0:
            # Loss
            return -1

        # Win
        return (
            1
            + view_game_state_next.character.health_current
            / view_game_state_next.character.health_max
        )

    # Floor climb
    diff_floor = 0
    if (
        view_game_state.map.y_current is not None
        and view_game_state_next.map.y_current is not None
    ):
        diff_floor = view_game_state_next.map.y_current - view_game_state.map.y_current

    # Upgrades
    num_upgrades = sum([view_card.name.endswith("+") for view_card in view_game_state.deck])
    num_upgrades_next = sum(
        [view_card.name.endswith("+") for view_card in view_game_state_next.deck]
    )
    diff_upgrades = num_upgrades_next - num_upgrades

    # Health difference
    diff_health_char = (
        view_game_state_next.character.health_current - view_game_state.character.health_current
    )

    return (
        _WEIGHT_HEALTH_CHAR * diff_health_char
        + _WEIGHT_FLOOR * diff_floor
        + _WEIGHT_UPGRADE * diff_upgrades
        + _PENALTY
    )
