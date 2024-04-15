from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.card.base import BaseCardLogic


class StrikeLogic(BaseCardLogic):
    def use(self, target_monster_id: int) -> list[Effect]:
        return [Effect(context.get_char()[0], target_monster_id, EffectType.DAMAGE, 6)]
