from game.context import Context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.move.base import BaseMoveLogic


class AttackLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, context: Context, source_entity_id: int) -> list[Effect]:
        return [Effect(source_entity_id, context.get_char()[0], EffectType.DAMAGE, 6)]
