from abc import ABC, abstractmethod
from typing import Tuple, Optional

from game.battle.comm import ActionType
from game.battle.comm import BattleView


class BaseAgent(ABC):
    @abstractmethod
    def select_action(
        self, battle_view: BattleView
    ) -> Tuple[ActionType, Optional[int]]:
        raise NotImplementedError
