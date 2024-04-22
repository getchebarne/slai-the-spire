from abc import ABC, abstractmethod
from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect


class BaseCardLogic(ABC):
    @abstractmethod
    def use(self, context: Context, target_monster_id: Optional[int] = None) -> list[Effect]:
        raise NotImplementedError
