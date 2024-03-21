from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.move.base import BaseMoveLogic


class DefendLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, source_entity_id: int) -> list[Effect]:
        return [Effect(source_entity_id, source_entity_id, EffectType.BLOCK, 6)]
