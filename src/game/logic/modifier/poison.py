from typing import Optional

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.modifier.base import BaseModifierLogic


class PoisonLogic(BaseModifierLogic):
    def at_start_of_turn(self, source_entity_id: int, stacks: Optional[int]) -> list[Effect]:
        if stacks is None:
            raise ValueError(f"{self.__class__.__name__} requires a non-None stacks value")

        return [
            # TODO: this pierces block
            Effect(source_entity_id, source_entity_id, EffectType.DAMAGE, stacks)
        ]
