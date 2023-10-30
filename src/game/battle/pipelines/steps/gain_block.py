from game.battle.pipelines.steps.base import BaseStep
from game.effects.base import BaseEffect
from game.entities.actors.base import BaseActor


class GainBlock(BaseStep):
    def _apply_effect(self, effect: BaseEffect, target: BaseActor) -> None:
        target.block.current = min(
            target.block.max, target.block.current + effect.block
        )

    def _condition(self, effect: BaseEffect) -> bool:
        return bool(effect.block)
