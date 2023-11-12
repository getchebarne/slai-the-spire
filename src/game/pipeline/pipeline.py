from __future__ import annotations

from bisect import insort
from collections import deque
from typing import List

from game.effects.base import BaseEffect
from game.pipeline.steps.apply_str import ApplyStrength
from game.pipeline.steps.apply_weak import ApplyWeak
from game.pipeline.steps.base import AddTo
from game.pipeline.steps.base import BaseStep
from game.pipeline.steps.deal_damage import DealDamage
from game.pipeline.steps.gain_block import GainBlock
from game.pipeline.steps.gain_str import GainStrength
from game.pipeline.steps.gain_weak import GainWeak


DEFAULT_STEPS = {
    DealDamage(),
    GainBlock(),
    GainStrength(),
    ApplyStrength(),
    GainWeak(),
    ApplyWeak(),
}


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

    def __call__(self, effects: List[BaseEffect]) -> None:
        effect_queue = deque(effects)

        while effect_queue:
            effect = effect_queue.popleft()
            for step in self._steps:
                new_effect = step(effect)

                if new_effect is not None:
                    if new_effect.where == AddTo.BOT:
                        effect_queue.append(new_effect.effect)

                    elif new_effect.where == AddTo.TOP:
                        effect_queue.appendleft(new_effect.effect)

                    else:
                        raise ValueError(
                            f"Undefined `NewEffect` `where` attribute: {new_effect.where}"
                        )

    def __str__(self) -> str:
        return "\n".join([f"{step.priority} : {step.__class__.__name__}" for step in self._steps])
