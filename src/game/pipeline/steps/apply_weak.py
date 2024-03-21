from game.pipeline.steps.base import BaseStep


WEAK_FACTOR = 0.75


class ApplyWeak(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.damage = int(effect.damage * WEAK_FACTOR)

    def _condition(self, effect: BaseEffect) -> bool:
        return (
            effect.damage is not None
            and isinstance(effect, (CardEffect, MonsterEffect))
            and effect.source.modifiers.weak.stack.amount > 0
        )
