from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class GainBlock(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> NewEffects:
        target = effect.target

        target.block.current = min(target.block.max, target.block.current + effect.block)

        return NewEffects()

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.block is not None
