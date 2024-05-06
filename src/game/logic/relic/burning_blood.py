from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


HEAL = 6


class BurningBloodLogic(BaseRelicLogic):
    def battle_end(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return [Effect(EffectType.HEAL, HEAL, context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID)]
