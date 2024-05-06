from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.pipeline.steps.base import BaseStep


class GainStrength(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        # TODO: use defaultdict?
        context.entity_modifiers[(effect.target_entity_id, "Strength")] += effect.value

    def _condition(self, context: Context, effect: Effect) -> bool:
        return effect.type == EffectType.GAIN_STR
