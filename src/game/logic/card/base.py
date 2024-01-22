from abc import ABC, abstractmethod
from typing import Optional

from game.core.effect import Effect


class BaseCardLogic(ABC):
    @abstractmethod
    def use(self, monster_idx: Optional[int] = None) -> list[Effect]:
        raise NotImplementedError
