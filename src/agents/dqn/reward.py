from src.game.combat.view import CombatView


def compute_reward(
    combat_view_t: CombatView, combat_view_tp1: CombatView, game_over_flag: bool
) -> float:
    if game_over_flag:
        if combat_view_tp1.character.health.current <= 0:
            # Loss
            return -1

        # Win
        return combat_view_tp1.character.health.current / combat_view_tp1.character.health.max

    # Card diff
    weight_cards_in_hand = 0.0075
    diff_cards_in_hand = weight_cards_in_hand * (
        len(combat_view_tp1.hand) - len(combat_view_t.hand)
    )

    # Energy diff
    weight_energy = 0.0075
    diff_energy = weight_energy * (combat_view_tp1.energy.current - combat_view_t.energy.current)

    # Damage
    # weight_damage = 0.0375
    # diff_damage = weight_damage * (
    #     combat_view_tp1.monsters[0].health.current - combat_view_t.monsters[0].health.current
    # )

    # Health
    diff_health = (
        combat_view_tp1.character.health.current - combat_view_t.character.health.current
    ) / combat_view_t.character.health.max

    # Still going
    return diff_health + diff_cards_in_hand + diff_energy
