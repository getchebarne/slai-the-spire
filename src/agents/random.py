import random
from typing import Optional

from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView


class RandomAgent(BaseAgent):
    def select_action(self, combat_view: CombatView) -> Optional[Action]:
        if not (
            any([card.can_be_selected for card in combat_view.hand])
            or any([monster.can_be_selected for monster in combat_view.monsters])
        ):
            # TODO: fix
            return None

        if combat_view.effect is not None:
            # TODO: only applies to survivor's discard for now, fix
            return Action(
                ActionType.SELECT_CARD,
                random.choice(
                    [card for card in combat_view.hand if card.can_be_selected]
                ).entity_id,
            )

        # If there's no more energy, end the turn
        if combat_view.energy.current == 0:
            return Action(ActionType.END_TURN)

        if not any([card.is_selected for card in combat_view.hand]):
            # Select random card in hand
            return Action(ActionType.SELECT_CARD, random.choice(combat_view.hand).entity_id)

        if not any([monster.can_be_selected for monster in combat_view.monsters]):
            # Confirm the selected card
            return Action(ActionType.CONFIRM)

        # Select random monster
        return Action(ActionType.SELECT_MONSTER, random.choice(combat_view.monsters).entity_id)
