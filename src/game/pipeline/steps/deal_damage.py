from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class DealDamage(BaseStep):
    def _apply_effect(self, effect: Effect) -> NewEffects:
        damage = effect.value
        target = effect.target

        # Remove block
        dmg_over_block = max(0, damage - target.block)
        target.block = max(0, target.block - damage)

        # Remove health
        target.current_health = max(0, target.current_health - dmg_over_block)

        return NewEffects()

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.DAMAGE
