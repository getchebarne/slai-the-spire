from game.context import Entity
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.moves.base import BaseMoveLogic


class Defend(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, owner: Entity) -> list[Effect]:
        return [Effect(EffectType.DAMAGE, 6, owner, owner)]
