from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.base import NewEffects


class GainBlock(BaseStep):
    def _apply_effect(self, effect: Effect) -> NewEffects:
        target = effect.target

        # TODO: parametrize max block
        target.block.current = min(999, target.block.current + effect.value)

        return NewEffects()

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.BLOCK
