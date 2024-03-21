from abc import ABC, abstractmethod

from game.core.effect import Effect


class BaseMoveLogic(ABC):
    @abstractmethod
    def use(self, source_entity_id: int) -> list[Effect]:
        raise NotImplementedError
