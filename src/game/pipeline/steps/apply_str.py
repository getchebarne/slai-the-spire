from game.core.effect import Effect
from game.core.effect import EffectType
from game import context
from game.pipeline.steps.base import BaseStep


class ApplyStrength(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        effect.value += context.entity_modifiers[(effect.source_entity_id, "Strength")]

    def _condition(self, effect: Effect) -> bool:
        return (
            effect.type == EffectType.DAMAGE
            and (effect.source_entity_id, "Strength") in context.entity_modifiers
        )
