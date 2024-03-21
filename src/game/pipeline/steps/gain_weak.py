from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep


class GainWeak(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.target.modifiers.weak.stack.increase(effect.weak)

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.weak is not None
