from abc import ABC, abstractmethod

from src.game.battle.engine import Action


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self) -> Action:
        raise NotImplementedError
