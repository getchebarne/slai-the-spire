from __future__ import annotations
from bisect import insort
from typing import List

from game.battle.pipeline.steps.base import BaseStep
from game.battle.pipeline.steps.deal_damage import DealDamage
from game.battle.pipeline.steps.gain_block import GainBlock
from game.effects.base import BaseEffect


DEFAULT_STEPS = {DealDamage(), GainBlock()}


class EffectPipeline:
    def __init__(self):
        # Initialize w/ default steps. TODO: can this be improved?
        self._steps: List[BaseStep] = []

        for step in DEFAULT_STEPS:
            self.add_step(step)

    def add_step(self, step: BaseStep) -> None:
        if any(other.priority == step.priority for other in self._steps):
            raise ValueError(f"A step with priority {step.priority} already exists.")

        insort(self._steps, step)

    def __call__(self, effect: BaseEffect) -> None:
        for step in self._steps:
            step(effect)
