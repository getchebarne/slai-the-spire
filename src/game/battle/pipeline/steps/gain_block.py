from game.battle.pipeline.steps.base import BaseStep
from game.effects.base import BaseEffect


class GainBlock(BaseStep):
    @property
    def priority(self) -> int:
        return 1

    def _apply_effect(self, effect: BaseEffect) -> None:
        target = effect.target

        target.block.current = min(
            target.block.max, target.block.current + effect.block
        )

    def _condition(self, effect: BaseEffect) -> bool:
        return bool(effect.block)
