from abc import ABC, abstractmethod
from typing import Tuple, Optional

from game.battle.engine import ActionType
from game.battle.engine import BattleView


class BaseAgent(ABC):
    @abstractmethod
    def select_action(
        self, battle_view: BattleView
    ) -> Tuple[ActionType, Optional[int]]:
        raise NotImplementedError
