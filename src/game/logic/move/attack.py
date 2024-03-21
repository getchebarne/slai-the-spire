from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.move.base import BaseMoveLogic


class AttackLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, source_entity_id: int) -> list[Effect]:
        return [Effect(source_entity_id, context.char_entity_id(), EffectType.DAMAGE, 6)]
