from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.move.base import BaseMoveLogic


class AttackLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, source_entity_id: int) -> list[Effect]:
        return [Effect(source_entity_id, context.get_char()[0], EffectType.DAMAGE, 6)]
