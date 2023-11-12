from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep


class GainStrength(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.target.modifiers.strength.stack.increase(effect.plus_str)

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.plus_str is not None
