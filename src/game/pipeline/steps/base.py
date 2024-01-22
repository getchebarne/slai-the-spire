from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from game.core.effect import Effect


# TODO: review if this is the best solution
@dataclass
class NewEffects:
    add_to_bot: List[Effect] = field(default_factory=list)
    add_to_top: List[Effect] = field(default_factory=list)


class BaseStep(ABC):
    def __call__(self, effect: Effect) -> NewEffects:
        if self._condition(effect):
            return self._apply_effect(effect)

        return NewEffects()

    @abstractmethod
    def _apply_effect(self, effect: Effect) -> NewEffects:
        raise NotImplementedError

    @abstractmethod
    def _condition(self, effect: Effect) -> bool:
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
