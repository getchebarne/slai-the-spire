from typing import Optional

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.card.base import BaseCardLogic


BLOCK = 5


class DefendLogic(BaseCardLogic):
    def use(self, context: Context, target_monster_id: Optional[int] = None) -> list[Effect]:
        return [Effect(EffectType.BLOCK, BLOCK, context.CHAR_ENTITY_ID, context.CHAR_ENTITY_ID)]
