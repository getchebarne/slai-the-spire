from game.battle.pipelines.steps.base import BaseStep
from game.effects.base import BaseEffect


class DealDamage(BaseStep):
    @property
    def priority(self) -> int:
        return 0

    def _apply_effect(self, effect: BaseEffect) -> None:
        damage = effect.damage
        target = effect.target

        # Remove block
        dmg_over_block = max(0, damage - target.block.current)
        target.block.current = max(0, target.block.current - damage)

        # Remove health
        target.health.current = max(0, target.health.current - dmg_over_block)

    def _condition(self, effect: BaseEffect) -> bool:
        return bool(effect.damage)
