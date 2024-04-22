from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.card.base import BaseCardLogic


DAMAGE = 3
WEAK = 1


class NeutralizeLogic(BaseCardLogic):
    def use(self, context: Context, target_monster_id: int) -> list[Effect]:
        return [
            Effect(context.get_char()[0], target_monster_id, EffectType.DAMAGE, DAMAGE),
            Effect(context.get_char()[0], target_monster_id, EffectType.WEAK, WEAK),
        ]
