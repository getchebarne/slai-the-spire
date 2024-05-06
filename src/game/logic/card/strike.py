from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.logic.card.base import BaseCardLogic


DAMAGE = 6


class StrikeLogic(BaseCardLogic):
    def use(self, context: Context, target_monster_id: int) -> list[Effect]:
        return [Effect(EffectType.DAMAGE, DAMAGE, context.CHAR_ENTITY_ID, target_monster_id)]
