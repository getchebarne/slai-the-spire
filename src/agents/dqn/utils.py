from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.state import State
from src.game.combat.view import CombatView


# TODO: these should be more "global", i.e., not specific to the DQN agent
def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    # Cards in hand
    valid_action_mask = [
        card.entity_id in combat_view.entity_selectable_ids for card in combat_view.hand
    ] + [False] * (MAX_HAND_SIZE - len(combat_view.hand))

    # Monsters
    valid_action_mask += [
        monster.entity_id in combat_view.entity_selectable_ids for monster in combat_view.monsters
    ] + [False] * (MAX_MONSTERS - len(combat_view.monsters))

    # End turn
    valid_action_mask.append(combat_view.state == State.DEFAULT)

    return valid_action_mask


def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 5:
        return Action(ActionType.SELECT_ENTITY, combat_view.hand[action_idx].entity_id)

    if action_idx == 5:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[5 - action_idx].entity_id)

    if action_idx == 6:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


def action_to_action_idx(action: Action, combat_view: CombatView) -> int:
    if action.type == ActionType.END_TURN:
        return 6

    for idx, card in enumerate(combat_view.hand):
        if card.entity_id == action.target_id:
            return idx

    for idx, monster in enumerate(combat_view.monsters):
        if monster.entity_id == action.target_id:
            return idx + 5

    raise ValueError(f"Unable to convert action: {action}")
