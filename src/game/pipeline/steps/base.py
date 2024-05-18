from __future__ import annotations

from abc import ABC, abstractmethod

from src.game.core.effect import Effect
from src.game.ecs.manager import ECSManager


class BaseStep(ABC):
    def __call__(
        self, manager: ECSManager, target_entity_id: int, effect: Effect
    ) -> tuple[list[Effect], list[Effect]]:
        if self._condition(effect):
            # Apply effect
            self._apply_effect(manager, target_entity_id, effect)

            # Return new effects. TODO: reenable
            # return (
            #     self._add_to_bot_effects(effect),
            #     self._add_to_top_effects(effect),
            # )

        # Otherwise, return empty lists
        return [], []

    @abstractmethod
    def _apply_effect(self, manager: ECSManager, target_entity_id: int, effect: Effect) -> None:
        raise NotImplementedError

    @abstractmethod
    def _condition(self, target_entity_id: int, effect: Effect) -> bool:
        raise NotImplementedError

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
