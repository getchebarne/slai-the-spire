from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.view import CombatView


def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    # Cards in hand
    valid_action_mask = [card.is_selectable for card in combat_view.hand] + [False] * (
        MAX_HAND_SIZE - len(combat_view.hand)
    )

    # Monsters
    valid_action_mask += [monster.is_selectable for monster in combat_view.monsters] + [False] * (
        MAX_MONSTERS - len(combat_view.monsters)
    )

    # End turn
    valid_action_mask.append(True)

    return valid_action_mask
