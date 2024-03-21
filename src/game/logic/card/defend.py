from typing import Optional

from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class DefendLogic(BaseCardLogic):
    def use(self, target_monster_id: Optional[int] = None) -> list[Effect]:
        return [Effect(context.char_entity_id(), context.char_entity_id(), EffectType.BLOCK, 5)]
