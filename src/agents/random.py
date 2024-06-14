import random
from typing import Optional

from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView


class RandomAgent(BaseAgent):
    def select_action(self, combat_view: CombatView) -> Optional[Action]:
        # If there's a card selected
        if any([card.is_selected for card in combat_view.hand]):

            # Check if monsters can be selected
            monsters_can_be_selected = [
                monster for monster in combat_view.monsters if monster.can_be_selected
            ]
            if len(monsters_can_be_selected) > 0:
                # If so, select one randomly
                return Action(
                    ActionType.SELECT_ENTITY, random.choice(monsters_can_be_selected).entity_id
                )

            # Else, confirm the selected card
            return Action(ActionType.CONFIRM)

        # Otherwise, if a card can be selected, select one randomly
        cards_can_be_selected = [card for card in combat_view.hand if card.can_be_selected]
        if len(cards_can_be_selected) > 0:
            return Action(ActionType.SELECT_ENTITY, random.choice(cards_can_be_selected).entity_id)

        # If there's an effect pending input targets, select its targets randomly
        # TODO: improve this
        if combat_view.effect is not None:
            if combat_view.effect.name == "Discard":
                return Action(
                    ActionType.SELECT_ENTITY,
                    random.choice(
                        [card for card in combat_view.hand if card.can_be_selected]
                    ).entity_id,
                )

            raise ValueError(f"Unsupported effect {combat_view.effect.name}")

        # At this point, there's no currently selected cards, no cards can be selected, and
        # there's no effects pending input targets. The agent can only end the turn
        return Action(ActionType.END_TURN)
