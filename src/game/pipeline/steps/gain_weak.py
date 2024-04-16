from game.core.effect import Effect
from game import context
from game.pipeline.steps.base import BaseStep


class GainWeak(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        # TODO: use defaultdict
        if (effect.source_entity_id, "Weak") in context.entity_modifiers:
            context.entity_modifiers[(effect.source_entity_id, "Weak")] += effect.value
        else:
            context.entity_modifiers[(effect.source_entity_id, "Weak")] = effect.value

    def _condition(self, effect: Effect) -> bool:
        return True
