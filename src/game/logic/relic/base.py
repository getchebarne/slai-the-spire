from abc import ABC
from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect


class BaseRelicLogic(ABC):
    def at_start_of_turn(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def at_end_of_turn(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def at_start_of_battle(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []

    def at_end_of_battle(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return []
