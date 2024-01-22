from typing import Optional

from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class DefendLogic(BaseCardLogic):
    def use(self, monster_idx: Optional[int] = None) -> list[Effect]:
        return [Effect(EffectType.BLOCK, 5, context.char, context.char)]
