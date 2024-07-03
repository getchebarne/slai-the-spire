import random

from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView


class RandomAgent(BaseAgent):
    def select_action(self, combat_view: CombatView) -> Action:
        # Check if there's an active card
        if any([card.is_active for card in combat_view.hand]):
            return Action(
                ActionType.SELECT_ENTITY,
                random.choice([monster.entity_id for monster in combat_view.monsters]),
            )

        # Else, select a card
        card_is_selectable_ids = [
            card.entity_id for card in combat_view.hand if card.cost <= combat_view.energy.current
        ]
        if card_is_selectable_ids:
            return Action(ActionType.SELECT_ENTITY, random.choice(card_is_selectable_ids))

        # Else, end turn
        return Action(ActionType.END_TURN)
