from src.game.combat.view import CombatView


def compute_reward(combat_view: CombatView, game_over_flag: bool) -> float:
    if not game_over_flag:
        # Still going
        return 0

    if combat_view.character.health.current <= 0:
        # Loss
        return -1

    # Win TODO: the agent may gain health, think about this
    return combat_view.character.health.current / combat_view.character.health.max
