from typing import Optional

from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class NeutralizeLogic(BaseCardLogic):
    def use(self, monster_idx: Optional[int] = None) -> list[Effect]:
        monster = context.monsters[monster_idx]
        return [
            Effect(EffectType.DAMAGE, 3, context.char, monster),
            Effect(EffectType.WEAK, 1, context.char, monster),
        ]
