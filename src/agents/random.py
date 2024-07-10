import random

from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView
from src.game.combat.view.state import StateView


# from src.game.combat.view.effect import EffectViewType


class RandomAgent(BaseAgent):
    def select_action(self, combat_view: CombatView) -> Action:
        if combat_view.state == StateView.AWAIT_EFFECT_TARGET:
            # TODO: select based on target type
            return Action(ActionType.SELECT_ENTITY, random.choice(combat_view.hand).entity_id)

        if combat_view.state == StateView.DEFAULT:
            # TODO: set entitiy selectable ids, this shouldn't be done here
            card_is_selectable_ids = [
                card.entity_id
                for card in combat_view.hand
                if card.cost <= combat_view.energy.current
            ]
            if card_is_selectable_ids:
                return Action(ActionType.SELECT_ENTITY, random.choice(card_is_selectable_ids))

        if combat_view.state == StateView.AWAIT_CARD_TARGET:
            # Select random monster
            return Action(ActionType.SELECT_ENTITY, random.choice(combat_view.monsters).entity_id)

        return Action(ActionType.END_TURN)
