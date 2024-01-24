from abc import ABC, abstractmethod

from game.core.effect import Effect
from game.core.monster import Monster


class BaseMoveLogic(ABC):
    @abstractmethod
    def use(self, source: Monster) -> list[Effect]:
        raise NotImplementedError
