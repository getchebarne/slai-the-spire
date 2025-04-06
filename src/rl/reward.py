from src.game.combat.view import CombatView


WEIGHT_CARDS_IN_HAND = 0.025
WEIGHT_ENERGY = 0.025
WEIGHT_HEALTH = 3 * (WEIGHT_CARDS_IN_HAND + WEIGHT_ENERGY)


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

    diff_cards_in_hand = len(combat_view_next.hand) - len(combat_view.hand)
    diff_energy = combat_view_next.energy.current - combat_view.energy.current
    diff_health_char = (
        combat_view_next.character.health_current - combat_view.character.health_current
    )

    return (
        WEIGHT_CARDS_IN_HAND * diff_cards_in_hand
        + WEIGHT_ENERGY * diff_energy
        + WEIGHT_HEALTH * diff_health_char
    )
