from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.view import CombatView


# TODO: must change when more monsters are added
# TODO: make more readable
def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    if any(card.is_active for card in combat_view.hand):
        # Only valid action is to select the monster
        return [False] * (2 * MAX_HAND_SIZE) + [True, False]

    if combat_view.effect is not None:
        # TODO: only contemplating EffectType.DISCARD for now
        valid_action_mask = [False] * MAX_HAND_SIZE
        valid_action_mask.extend([True] * len(combat_view.hand))
        valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
        valid_action_mask.extend([False, False])

        return valid_action_mask

    valid_action_mask = [card.cost <= combat_view.energy.current for card in combat_view.hand]
    valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
    valid_action_mask.extend([False] * MAX_HAND_SIZE)
    valid_action_mask.extend([False, True])

    return valid_action_mask


# TODO: adapt for multiple monsters
# TODO: adapt for multiple halted effects
def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 2 * MAX_HAND_SIZE:
        return Action(
            ActionType.SELECT_ENTITY, combat_view.hand[action_idx % MAX_HAND_SIZE].entity_id
        )

    if action_idx == 2 * MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if action_idx == 2 * MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")
