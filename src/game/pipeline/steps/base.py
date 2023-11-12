from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from game.effects.base import BaseEffect


class AddTo(Enum):
    TOP = 0
    BOT = 1


# TODO: review is this is the best solution
@dataclass
class NewEffect:
    effect: BaseEffect
    where: AddTo


class BaseStep(ABC):
    def __call__(self, effect: BaseEffect) -> Optional[NewEffect]:
        if self._condition(effect):
            return self._apply_effect(effect)

    @abstractmethod
    def _apply_effect(self, effect: BaseEffect) -> Optional[NewEffect]:
        raise NotImplementedError

    @abstractmethod
    def _condition(self, effect: BaseEffect) -> bool:
        raise NotImplementedError

    @property
    def priority(self) -> int:
        from game.pipeline.steps.order import STEP_ORDER

        return STEP_ORDER.index(self.__class__)

    def __lt__(self, other: BaseStep) -> bool:
        if not isinstance(other, BaseStep):
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

        return self.priority < other.priority
