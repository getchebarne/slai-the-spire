from src.game.combat.view import CombatView


WEIGHT_CARDS_IN_HAND = 0.025
WEIGHT_ENERGY = 0.025
WEIGHT_HEALTH = 3 * (WEIGHT_CARDS_IN_HAND + WEIGHT_ENERGY)


def compute_reward(
    combat_view_t: CombatView, combat_view_tp1: CombatView, game_over_flag: bool
) -> float:
    health_current = combat_view_tp1.character.health_current
    health_max = combat_view_tp1.character.health_max
    if game_over_flag:
        if health_current <= 0:
            # Loss
            return -1

        # Win
        return 1 + health_current / health_max

    diff_cards_in_hand = len(combat_view_tp1.hand) - len(combat_view_t.hand)
    diff_energy = combat_view_tp1.energy.current - combat_view_t.energy.current
    diff_health_char = (
        combat_view_tp1.character.health_current - combat_view_t.character.health_current
    )

    return (
        WEIGHT_CARDS_IN_HAND * diff_cards_in_hand
        + WEIGHT_ENERGY * diff_energy
        + WEIGHT_HEALTH * diff_health_char
    )
