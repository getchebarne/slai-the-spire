from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.pipeline.steps.base import BaseStep


WEAK_FACTOR = 0.75


class ApplyWeak(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        effect.value = int(effect.damage * WEAK_FACTOR)

    def _condition(self, context: Context, effect: Effect) -> bool:
        return (
            effect.type == EffectType.DAMAGE
            and (effect.source_entity_id, "Weak") in context.entity_modifiers
        )
