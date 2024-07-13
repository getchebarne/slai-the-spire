import random

from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView


class RandomAgent(BaseAgent):
    def select_action(self, combat_view: CombatView) -> Action:
        if combat_view.entity_selectable_ids:
            return Action(
                ActionType.SELECT_ENTITY, random.choice(combat_view.entity_selectable_ids)
            )

        return Action(ActionType.END_TURN)
