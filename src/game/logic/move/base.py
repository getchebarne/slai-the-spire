from abc import ABC, abstractmethod

from game.core.effect import Effect
from game.context import Context


class BaseMoveLogic(ABC):
    @abstractmethod
    def use(self, context: Context, source_entity_id: int) -> list[Effect]:
        raise NotImplementedError
