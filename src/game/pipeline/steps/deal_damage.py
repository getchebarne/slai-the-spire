from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class DealDamage(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        damage = effect.damage
        target = effect.target

        # Remove block
        dmg_over_block = max(0, damage - target.block.current)
        target.block.current = max(0, target.block.current - damage)

        # Remove health
        target.health.current = max(0, target.health.current - dmg_over_block)

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.damage is not None
