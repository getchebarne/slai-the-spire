from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.view import CombatView


# TODO: these should be more "global", i.e., not specific to the DQN agent
# TODO: make more legible
def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    if any([card_view.is_active for card_view in combat_view.hand]):
        return [False] * MAX_HAND_SIZE + [True] * MAX_MONSTERS + [False]

    valid_action_mask = []
    for card_view in combat_view.hand:
        if card_view.cost <= combat_view.energy.current:
            valid_action_mask.append(True)

        else:
            valid_action_mask.append(False)

    return valid_action_mask + [False] * (MAX_HAND_SIZE - len(combat_view.hand)) + [False] + [True]


# TODO: adapt for multiple monsters
# TODO: adapt for discard
def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.hand[action_idx].entity_id)

    if action_idx == MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if action_idx == MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")
