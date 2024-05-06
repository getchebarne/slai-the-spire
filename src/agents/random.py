import numpy as np

from src.agents.base import BaseAgent
from src.game.battle.engine import Action
from src.game.battle.engine import ActionType
from src.game.context import BattleState
from src.game.context import Context


class RandomAgent(BaseAgent):
    # TODO: choose more actions other than card index
    def select_action(self, context: Context) -> Action:
        if context.state == BattleState.DEFAULT:
            if context.energy.current == 0:
                # TODO: play `Neutralize` if it's in hand
                return Action(ActionType.END_TURN, None)

            card_idx = np.random.choice(range(len(context.hand)))
            return Action(ActionType.SELECT_CARD, card_idx)

        if context.state == BattleState.AWAIT_TARGET:
            monster_entity_id = np.random.choice(
                [monster_id for monster_id, _ in context.get_monster_data()]
            )
            return Action(ActionType.SELECT_TARGET, monster_entity_id)
