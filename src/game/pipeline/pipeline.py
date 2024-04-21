from __future__ import annotations

from bisect import insort
from collections import deque
from typing import List

from src.game.core.effect import Effect
from src.game.context import Context
from src.game.pipeline.steps.base import BaseStep
from src.game.pipeline.steps.deal_damage import DealDamage
from src.game.pipeline.steps.gain_block import GainBlock


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

    def __call__(self, context: Context, effects: List[Effect]) -> None:
        effect_queue = deque(effects)

        while effect_queue:
            effect = effect_queue.popleft()
            for step in self._steps:
                add_to_bot_effects, add_to_top_effects = step(context, effect)

                # Add new effects to the queue
                effect_queue.extend(add_to_bot_effects)
                effect_queue.extendleft(add_to_top_effects)

    def __str__(self) -> str:
        return "\n".join([f"{step.priority} : {step.__class__.__name__}" for step in self._steps])
