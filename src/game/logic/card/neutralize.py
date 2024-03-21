from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class NeutralizeLogic(BaseCardLogic):
    def use(self, target_monster_id: int) -> list[Effect]:
        return [
            Effect(context.char_entity_id(), target_monster_id, EffectType.DAMAGE, 3),
            Effect(context.char_entity_id(), target_monster_id, EffectType.WEAK, 1),
        ]
