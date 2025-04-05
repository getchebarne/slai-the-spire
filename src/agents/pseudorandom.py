import random

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView


def select_action(combat_view: CombatView) -> Action:
    if any([card.is_active for card in combat_view.hand]):
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if combat_view.effect is not None:
        return Action(
            ActionType.SELECT_ENTITY, random.choice([card.entity_id for card in combat_view.hand])
        )

    id_selectable_cards = [
        card.entity_id for card in combat_view.hand if card.cost <= combat_view.energy.current
    ]
    if id_selectable_cards:
        return Action(ActionType.SELECT_ENTITY, random.choice(id_selectable_cards))

    return Action(ActionType.END_TURN)
