from typing import Optional

from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class StrikeLogic(BaseCardLogic):
    def use(self, monster_idx: Optional[int] = None) -> list[Effect]:
        return [Effect(EffectType.DAMAGE, 6, context.char, context.monsters[monster_idx])]
