from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.core.monster import Monster
from game.logic.move.base import BaseMoveLogic


class AttackLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, source: Monster) -> list[Effect]:
        return [Effect(EffectType.DAMAGE, 6, source, context.char)]
