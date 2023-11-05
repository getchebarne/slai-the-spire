from typing import List

from game.battle.pipeline.steps.base import BaseStep
from game.battle.pipeline.steps.deal_damage import DealDamage
from game.battle.pipeline.steps.gain_block import GainBlock
from game.effects.base import BaseEffect


class EffectPipeline:
    _steps: List[BaseStep] = [DealDamage(), GainBlock()]

    def __call__(self, effect: BaseEffect) -> None:
        for step in self._steps:
            # TODO: systems can create additional effects to be processed
            step(effect)
