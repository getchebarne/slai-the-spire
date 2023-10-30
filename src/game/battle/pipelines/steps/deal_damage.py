from game.battle.pipelines.steps.base import BaseStep
from game.effects.base import BaseEffect
from game.entities.actors.base import BaseActor


class DealDamage(BaseStep):
    def _apply_effect(self, effect: BaseEffect, target: BaseActor) -> None:
        damage = effect.damage

        # Remove block
        dmg_over_block = max(0, damage - target.block.current)
        target.block.current = max(0, target.block.current - damage)

        # Remove health
        target.health.current = max(0, target.health.current - dmg_over_block)

    def _condition(self, effect: BaseEffect) -> bool:
        return bool(effect.damage)
