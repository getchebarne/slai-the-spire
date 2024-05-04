from abc import ABC, abstractmethod

from src.game.battle.engine import Action
from src.game.context import Context


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, context: Context) -> Action:
        raise NotImplementedError
