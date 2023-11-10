from game.effects.base import BaseEffect
from game.effects.card import CardEffect
from game.pipeline.steps.base import BaseStep


class ApplyStrength(BaseStep):
    @property
    def priority(self) -> int:
        return 0

    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.damage += effect.source.buffs.strength.stack_amount

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.damage is not None and isinstance(effect, CardEffect)
