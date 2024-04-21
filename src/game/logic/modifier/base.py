from abc import ABC
from typing import Optional

from src.game.core.effect import Effect
from src.game.core.entity import Entity


class BaseModifierLogic(ABC):
    def at_start_of_turn(self, source_entity: Entity, stacks: Optional[int]) -> list[Effect]:
        return []

    def at_end_of_turn(self, source_entity: Entity, stacks: Optional[int]) -> list[Effect]:
        return []

    def at_end_of_battle(self, source_entity: Entity, stacks: Optional[int]) -> list[Effect]:
        return []
