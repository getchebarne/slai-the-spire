from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class GainWeak(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        effect.target.modifiers.weak.stack.increase(effect.weak)

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.weak is not None
