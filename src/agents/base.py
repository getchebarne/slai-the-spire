from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

from game.battle.engine import ActionType
from game.battle.engine import BattleView


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, battle_view: BattleView) -> Tuple[ActionType, Optional[int]]:
        raise NotImplementedError
