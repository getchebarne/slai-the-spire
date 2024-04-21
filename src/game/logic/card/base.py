from abc import ABC, abstractmethod
from typing import Optional

from src.game.core.effect import Effect
from src.game.context import Context


class BaseCardLogic(ABC):
    @abstractmethod
    def use(self, context: Context, target_monster_id: Optional[int] = None) -> list[Effect]:
        raise NotImplementedError
