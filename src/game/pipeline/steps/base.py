from __future__ import annotations

from abc import ABC, abstractmethod

from src.game.core.effect import Effect
from src.game.context import Context


class BaseStep(ABC):
    def __call__(self, context: Context, effect: Effect) -> tuple[list[Effect], list[Effect]]:
        if self._condition(context, effect):
            # Apply effect
            self._apply_effect(context, effect)

            # Return new effects
            return (
                self._add_to_bot_effects(effect),
                self._add_to_top_effects(effect),
            )

        # Otherwise, return empty lists
        return [], []

    @abstractmethod
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        raise NotImplementedError

    @abstractmethod
    def _condition(self, context: Context, effect: Effect) -> bool:
        raise NotImplementedError

    def _add_to_bot_effects(self, effect: Effect) -> list[Effect]:
        return []

    def _add_to_top_effects(self, effect: Effect) -> list[Effect]:
        return []

    @property
    def priority(self) -> int:
        from src.game.pipeline.steps.order import STEP_ORDER

        return STEP_ORDER.index(self.__class__)

    def __lt__(self, other: BaseStep) -> bool:
        if not isinstance(other, BaseStep):
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

        return self.priority < other.priority
