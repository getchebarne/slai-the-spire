from src.game.combat.view import CombatView


def compute_reward(
    combat_view_t: CombatView, combat_view_tp1: CombatView, game_over_flag: bool
) -> float:
    if game_over_flag:
        return combat_view_tp1.character.health.current / combat_view_tp1.character.health.max

    # Still going
    return (
        combat_view_tp1.character.health.current - combat_view_t.character.health.current
    ) / combat_view_tp1.character.health.max
