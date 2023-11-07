import numpy as np

from agents.base import BaseAgent
from game.battle.engine import Action
from game.battle.engine import ActionType
from game.battle.engine import BattleState
from game.battle.engine import BattleView


class RandomAgent(BaseAgent):
    # TODO: choose more actions other than card index
    def select_action(self, battle_view: BattleView) -> Action:
        if battle_view.state == BattleState.DEFAULT:
            if battle_view.char.energy.current == 0:
                # TODO: play `Neutralize` if it's in hand
                return Action(ActionType.END_TURN, None)

            card_idx = np.random.choice(range(len(battle_view.hand)))
            return Action(ActionType.SELECT_CARD, card_idx)

        if battle_view.state == BattleState.AWAIT_TARGET:
            monster_idx = np.random.choice(range(len(battle_view.monsters)))
            return Action(ActionType.SELECT_TARGET, monster_idx)

            return Action(ActionType.SELECT_TARGET, None)
