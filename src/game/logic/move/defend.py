from game.core.effect import Effect
from game.core.effect import EffectType
from game.core.monster import Monster
from game.logic.move.base import BaseMoveLogic


class DefendLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, source: Monster) -> list[Effect]:
        return [Effect(EffectType.BLOCK, 6, source, source)]
