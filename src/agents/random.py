from typing import Tuple, Optional

import numpy as np

from agents.base import BaseAgent
from game.battle.comm import ActionType
from game.battle.comm import BattleView
from game.battle.state import BattleState


class RandomAgent(BaseAgent):
    # TODO: choose more actions other than card index
    def select_action(
        self, battle_view: BattleView
    ) -> Tuple[ActionType, Optional[int]]:
        if battle_view.state == BattleState.DEFAULT:
            if battle_view.char.energy.current == 0:
                # TODO: play `Neutralize` if it's in hand
                return ActionType.END_TURN, None

            card_idx = np.random.choice(range(len(battle_view.hand)))
            return ActionType.SELECT_CARD, card_idx

        if battle_view.state == BattleState.AWAIT_TARGET:
            if battle_view.active_card.requires_target:
                monster_idx = np.random.choice(range(len(battle_view.monsters)))
                return ActionType.SELECT_TARGET, monster_idx

            return ActionType.SELECT_TARGET, None
