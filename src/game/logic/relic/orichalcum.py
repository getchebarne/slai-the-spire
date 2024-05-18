from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


BLOCK = 6


class OrichalcumLogic(BaseRelicLogic):
    def char_turn_end(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        if context.entities[context.CHAR_ENTITY_ID].current_block == 0:
            return [
                Effect(EffectType.BLOCK, BLOCK, context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID)
            ]

        return []