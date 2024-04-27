from abc import ABC
from typing import Optional

from src.game.core.effect import Effect


class BaseModifierLogic(ABC):
    def at_start_of_turn(self, source_entity_id: int, stacks: Optional[int]) -> list[Effect]:
        return []

    def at_end_of_turn(self, source_entity_id: int, stacks: Optional[int]) -> list[Effect]:
        return []

    def at_end_of_battle(self, source_entity_id: int, stacks: Optional[int]) -> list[Effect]:
        return []
