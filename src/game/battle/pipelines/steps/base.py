from __future__ import annotations
from abc import ABC, abstractmethod

from game.effects.base import BaseEffect


class BaseStep(ABC):
    def __call__(self, effect: BaseEffect) -> None:
        if self._condition(effect):
            self._apply_effect(effect)

    @property
    @abstractmethod
    def priority(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _apply_effect(self, effect: BaseEffect) -> None:
        raise NotImplementedError

    @abstractmethod
    def _condition(self, effect: BaseEffect) -> bool:
        raise NotImplementedError

    def __lt__(self, other: BaseStep) -> bool:
        if not isinstance(other, BaseStep):
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

        return self.priority < other.priority
