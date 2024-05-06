from abc import ABC
from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect


class BaseRelicLogic(ABC):
    def char_turn_start(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def char_turn_end(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def battle_start(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def battle_end(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []
