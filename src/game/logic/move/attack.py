from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.move.base import BaseMoveLogic


DAMAGE = 6


class AttackLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, context: Context, source_entity_id: int) -> list[Effect]:
        return [Effect(EffectType.DAMAGE, DAMAGE, source_entity_id, context.CHAR_ENTITY_ID)]
