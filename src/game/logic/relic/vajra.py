from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


GAIN_STR = 1


class VajraLogic(BaseRelicLogic):
    def battle_start(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return [
            Effect(EffectType.GAIN_STR, GAIN_STR, context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID)
        ]
