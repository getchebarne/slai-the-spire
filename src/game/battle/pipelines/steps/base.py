from abc import ABC, abstractmethod

from game.effects.base import BaseEffect
from game.entities.actors.base import BaseActor


class BaseStep(ABC):
    def __call__(self, effect: BaseEffect, target: BaseActor) -> None:
        if self._condition(effect):
            self._apply_effect(effect, target)

    @abstractmethod
    def _apply_effect(self, effect: BaseEffect, target: BaseActor) -> None:
        raise NotImplementedError

    # TODO: should `target` also be an argument here?
    @abstractmethod
    def _condition(self, effect: BaseEffect) -> bool:
        raise NotImplementedError
