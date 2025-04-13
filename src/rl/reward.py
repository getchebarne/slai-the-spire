from src.game.combat.view import CombatView


WEIGHT_HEALTH_CHAR = 0.0250


def compute_reward(
    combat_view: CombatView, combat_view_next: CombatView, game_over_flag: bool
) -> float:
    if game_over_flag:
        if combat_view_next.character.health_current <= 0:
            # Loss
            return -1

        # Win
        return (
            1 + combat_view_next.character.health_current / combat_view_next.character.health_max
        )

    diff_health_char = (
        combat_view_next.character.health_current - combat_view.character.health_current
    )

    return WEIGHT_HEALTH_CHAR * diff_health_char
