from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.state import State
from src.game.combat.view import CombatView


# TODO: these should be more "global", i.e., not specific to the DQN agent
def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    if combat_view.state == State.DEFAULT:
        # Cards in hand
        valid_action_mask = [
            card.entity_id in combat_view.entity_selectable_ids for card in combat_view.hand
        ] + [False] * ((MAX_HAND_SIZE - len(combat_view.hand)) + MAX_HAND_SIZE)

    elif combat_view.state == State.AWAIT_EFFECT_TARGET:
        valid_action_mask = (
            [False] * MAX_HAND_SIZE
            + [card.entity_id in combat_view.entity_selectable_ids for card in combat_view.hand]
            + [False] * (MAX_HAND_SIZE - len(combat_view.hand))
        )

    elif combat_view.state == State.AWAIT_CARD_TARGET:
        valid_action_mask = [False] * 2 * MAX_HAND_SIZE

    else:
        raise ValueError

    # Monsters
    valid_action_mask += [
        monster.entity_id in combat_view.entity_selectable_ids for monster in combat_view.monsters
    ] + [False] * (MAX_MONSTERS - len(combat_view.monsters))

    # End turn
    valid_action_mask.append(combat_view.state == State.DEFAULT)

    return valid_action_mask


def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 2 * MAX_HAND_SIZE:
        return Action(
            ActionType.SELECT_ENTITY, combat_view.hand[action_idx % MAX_HAND_SIZE].entity_id
        )

    if action_idx == 2 * MAX_HAND_SIZE:
        return Action(
            ActionType.SELECT_ENTITY,
            combat_view.monsters[2 * MAX_HAND_SIZE - action_idx].entity_id,
        )

    if action_idx == 2 * MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


def action_to_action_idx(action: Action, combat_view: CombatView) -> int:
    if action.type == ActionType.END_TURN:
        return 2 * MAX_HAND_SIZE + 1

    for idx, card in enumerate(combat_view.hand):
        if card.entity_id == action.target_id:
            if combat_view.state == State.DEFAULT:
                return idx

            if combat_view.state == State.AWAIT_EFFECT_TARGET:
                return MAX_HAND_SIZE + idx

            raise ValueError(f"Selected card while in {combat_view.state}")

    for idx, monster in enumerate(combat_view.monsters):
        if monster.entity_id == action.target_id:
            return 2 * MAX_HAND_SIZE + idx

    raise ValueError(f"Unable to convert action: {action}")
