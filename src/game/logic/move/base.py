from abc import ABC, abstractmethod

from game.core.effect import Effect


class BaseMoveLogic(ABC):
    @abstractmethod
    def use(self) -> list[Effect]:
        raise NotImplementedError
