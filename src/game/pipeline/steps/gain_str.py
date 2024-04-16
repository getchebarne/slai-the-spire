from game.core.effect import Effect
from game import context
from game.pipeline.steps.base import BaseStep


class GainStrength(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        # TODO: use defaultdict?
        if (effect.source_entity_id, "Strength") in context.entity_modifiers:
            context.entity_modifiers[(effect.source_entity_id, "Strength")] += effect.value
        else:
            context.entity_modifiers[(effect.source_entity_id, "Strength")] = effect.value

    def _condition(self, effect: Effect) -> bool:
        return True
