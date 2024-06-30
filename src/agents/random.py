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
                ActionType.SELECT_MONSTER, random.choice(range(len(combat_view.monsters)))
            )

        # Else, select a card
        selectable_card_idxs = [
            idx
            for idx, card in enumerate(combat_view.hand)
            if card.cost <= combat_view.energy.current
        ]
        if selectable_card_idxs:
            return Action(ActionType.SELECT_CARD, random.choice(range(len(selectable_card_idxs))))

        # Else, end turn
        return Action(ActionType.END_TURN)
