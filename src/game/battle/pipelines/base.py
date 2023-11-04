from abc import ABC
from typing import List

from game.battle.pipelines.steps.base import BaseStep
from game.effects.base import BaseEffect


class BasePipeline(ABC):
    _steps: List[BaseStep] = []

    def __call__(self, effect: BaseEffect) -> None:
        for step in self._steps:
            # TODO: systems can create additional effects to be processed
            # TODO: will probably need to add `source` here
            step(effect)
