from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional


if TYPE_CHECKING:
    from game.effects.power import PowerEffect
    from game.entities.actors.base import BaseActor
    from game.entities.actors.characters.base import Character
    from game.entities.actors.monsters.group import MonsterGroup
    from game.pipeline.steps.base import BaseStep


DEFAULT_MIN = 0
DEFAULT_MAX = 999


@dataclass
class StackType:
    none: bool = False
    intensity: bool = False
    duration: bool = False
    counter: bool = False

    def __post_init__(self):
        if not any([self.none, self.intensity, self.duration, self.counter]):
            raise ValueError(
                "At least one of `none`, `intensity`, `duration`, or `counter` must be True"
            )


class BasePower(ABC):
    def __init__(self, stack_amount: int = 0):
        self.stack_amount = stack_amount

    @property
    @abstractmethod
    def stack_type(self) -> StackType:
        raise NotImplementedError

    @property
    def step(self) -> Optional[BaseStep]:
        return None

    @property
    def min(self) -> int:
        return DEFAULT_MIN

    @property
    def max(self) -> int:
        return DEFAULT_MAX

    def on_turn_end(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        return []

    def on_turn_start(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        return []

    def on_battle_end(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        return []

    def on_battle_start(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        return []

    def on_round_end(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        return []

    def on_round_start(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[PowerEffect]:
        if self.stack_type.duration:
            self.decrease_stack(1)

        return []

    def decrease_stack(self, amount: int) -> None:
        if self.stack_type.none:
            raise ValueError("Can't decrease the stack of a non-stackable power")

        self.stack_amount -= amount
        if self.stack_amount < self.min:
            raise ValueError(
                f"Power {self.__class__.__name__}'s stacks can't be decreased below {self.min}"
            )

    def increase_stack(self, amount: int) -> None:
        if self.stack_type.none:
            raise ValueError("Can't increase the stack of a non-stackable power")

        self.stack_amount += amount
        if self.stack_amount > self.max:
            raise ValueError(
                f"Power {self.__class__.__name__}'s stacks can't be increased above {self.max}"
            )
