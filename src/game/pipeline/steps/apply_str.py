from game.effects.base import BaseEffect
from game.effects.card import CardEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class ApplyStrength(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        effect.damage += effect.source.modifiers.strength.stack.amount

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.damage is not None and isinstance(effect, CardEffect)
