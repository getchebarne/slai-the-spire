from game.context import Context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class NeutralizeLogic(BaseCardLogic):
    def use(self, context: Context, target_monster_id: int) -> list[Effect]:
        return [
            Effect(context.get_char()[0], target_monster_id, EffectType.DAMAGE, 3),
            Effect(context.get_char()[0], target_monster_id, EffectType.WEAK, 1),
        ]
