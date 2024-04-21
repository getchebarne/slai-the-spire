from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.context import Context
from src.game.pipeline.steps.base import BaseStep


class ApplyStrength(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        effect.value += context.entity_modifiers[(effect.source_entity_id, "Strength")]

    def _condition(self, context: Context, effect: Effect) -> bool:
        return (
            effect.type == EffectType.DAMAGE
            and (effect.source_entity_id, "Strength") in context.entity_modifiers
        )
