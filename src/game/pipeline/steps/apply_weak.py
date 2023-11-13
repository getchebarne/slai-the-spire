from game.effects.base import BaseEffect
from game.effects.card import CardEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


WEAK_FACTOR = 0.75


class ApplyWeak(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        effect.damage = int(effect.damage * WEAK_FACTOR)

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return (
            effect.damage is not None
            and isinstance(effect, CardEffect)
            and effect.source.modifiers.weak.stack.amount > 0
        )
