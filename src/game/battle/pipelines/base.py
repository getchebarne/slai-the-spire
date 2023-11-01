from abc import ABC
from dataclasses import dataclass
from typing import List

from game.battle.pipelines.steps.base import BaseStep
from game.effects.base import BaseEffect
from game.entities.actors.base import BaseActor


@dataclass
class TargetedEffect:
    effect: BaseEffect
    source: BaseActor
    target: BaseActor


class BasePipeline(ABC):
    _steps: List[BaseStep] = []

    def __call__(self, targeted_effect: TargetedEffect) -> None:
        for step in self._steps:
            # TODO: systems can create additional effects to be processed
            # TODO: will probably need to add `source` here
            step(targeted_effect.effect, targeted_effect.target)
