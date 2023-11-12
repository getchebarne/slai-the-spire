from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep


class GainBlock(BaseStep):
    def _apply_effect(self, effect: BaseEffect) -> None:
        target = effect.target

        target.block.current = min(target.block.max, target.block.current + effect.block)

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.block is not None
