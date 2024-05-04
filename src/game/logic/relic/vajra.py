from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


PLUS_STR = 1


class VajraLogic(BaseRelicLogic):
    def at_start_of_battle(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return [
            Effect(context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID, EffectType.PLUS_STR, PLUS_STR)
        ]
