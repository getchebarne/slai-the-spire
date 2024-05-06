from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


DRAW = 2


class RingOfTheSnakeLogic(BaseRelicLogic):
    def battle_start(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return [Effect(EffectType.DRAW_CARD, DRAW, context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID)]
