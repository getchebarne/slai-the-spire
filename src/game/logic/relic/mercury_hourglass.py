from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.relic.base import BaseRelicLogic


DAMAGE = 3


class MercuryHourglassLogic(BaseRelicLogic):
    def char_turn_start(self, context: Context, count: Optional[int] = 0) -> list[Effect]:
        return [
            Effect(EffectType.DAMAGE, DAMAGE, context.CHAR_ENTITY_ID, monster_entity_id)
            for monster_entity_id, _ in context.get_monster_data()
        ]
