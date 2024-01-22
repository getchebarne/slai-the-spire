import numpy as np

from agents.base import BaseAgent
from game import context
from game.battle.engine import Action
from game.battle.engine import ActionType
from game.context import BattleState


class RandomAgent(BaseAgent):
    # TODO: choose more actions other than card index
    def select_action(self) -> Action:
        if context.state == BattleState.DEFAULT:
            if context.energy.current == 0:
                # TODO: play `Neutralize` if it's in hand
                return Action(ActionType.END_TURN, None)

            card_idx = np.random.choice(range(len(context.hand)))
            return Action(ActionType.SELECT_CARD, card_idx)

        if context.state == BattleState.AWAIT_TARGET:
            monster_idx = np.random.choice(range(len(context.monsters)))
            return Action(ActionType.SELECT_TARGET, monster_idx)
