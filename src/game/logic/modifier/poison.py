from game.core.effect import Effect
from game.core.effect import EffectType
from game.logic.modifier.base import BaseModifierLogic


class PoisonLogic(BaseModifierLogic):
    def at_start_of_turn(self, source_entity_id: int, stacks: int) -> list[Effect]:
        return [
            # TODO: this pierces block
            Effect(source_entity_id, source_entity_id, EffectType.DAMAGE, stacks),
            Effect(source_entity_id, source_entity_id, EffectType.POISON_DECREASE, 1),
        ]
