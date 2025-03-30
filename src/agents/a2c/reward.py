from src.game.combat.view import CombatView


def compute_reward(
    combat_view_t: CombatView, combat_view_tp1: CombatView, game_over_flag: bool
) -> float:
    if game_over_flag:
        if combat_view_tp1.character.health_current <= 0:
            # Loss
            return -1

        # Win
        return combat_view_tp1.character.health_current / combat_view_tp1.character.health_max

    # Still going. Penalize instant health loss
    diff_health = (
        combat_view_tp1.character.health_current - combat_view_t.character.health_current
    ) / combat_view_t.character.health_max

    # Add a small penalization term to discourage unnecessary actions
    return diff_health - 0.01
