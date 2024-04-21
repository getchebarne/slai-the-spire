from abc import ABC, abstractmethod

from src.game.core.effect import Effect
from src.game.context import Context


class BaseMoveLogic(ABC):
    @abstractmethod
    def use(self, context: Context, source_entity_id: int) -> list[Effect]:
        raise NotImplementedError
