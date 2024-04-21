from src.game.core.effect import Effect
from src.game.context import Context
from src.game.pipeline.steps.base import BaseStep


class GainStrength(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        # TODO: use defaultdict?
        if (effect.source_entity_id, "Strength") in context.entity_modifiers:
            context.entity_modifiers[(effect.source_entity_id, "Strength")] += effect.value
        else:
            context.entity_modifiers[(effect.source_entity_id, "Strength")] = effect.value

    def _condition(self, context: Context, effect: Effect) -> bool:
        return True
