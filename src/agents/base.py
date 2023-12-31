from abc import ABC, abstractmethod

from game.battle.engine import Action
from game.battle.engine import BattleView


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, battle_view: BattleView) -> Action:
        raise NotImplementedError
