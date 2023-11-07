from game.battle.pipeline.steps.base import BaseStep
from game.effects.base import BaseEffect
from game.effects.card import CardEffect


class ApplyStrength(BaseStep):
    @property
    def priority(self) -> int:
        return 0

    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.damage += effect.source.buffs.strength

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.damage and isinstance(effect, CardEffect)
