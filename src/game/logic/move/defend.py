from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.move.base import BaseMoveLogic


BLOCK = 6


class DefendLogic(BaseMoveLogic):
    # TODO: improve owner def
    def use(self, context: Context, source_entity_id: int) -> list[Effect]:
        return [Effect(source_entity_id, source_entity_id, EffectType.BLOCK, BLOCK)]
