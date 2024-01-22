from abc import ABC, abstractmethod

from game.battle.engine import Action


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self) -> Action:
        raise NotImplementedError
