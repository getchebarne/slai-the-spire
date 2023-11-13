from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class GainStrength(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        effect.target.modifiers.strength.stack.increase(effect.plus_str)

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.plus_str is not None
