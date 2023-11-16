from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional


if TYPE_CHECKING:
    from game.effects.modifier import ModifierEffect
    from game.entities.actors.base import BaseActor
    from game.entities.actors.characters.base import Character
    from game.entities.actors.monsters.group import MonsterGroup
    from game.pipeline.steps.base import BaseStep


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


@dataclass
class Stack:
    type: StackType
    min_amount: int
    max_amount: int
    amount: int = 0

    def decrease(self, value: int) -> None:
        if self.type.none:
            raise ValueError("Can't decrease the stack of a non-stackable modifier")

        self.amount -= value
        if self.amount < self.min_amount:
            raise ValueError(f"Stacks of {self} can't be decreased below {self.min_amount}")

    def increase(self, value: int) -> None:
        if self.type.none:
            raise ValueError("Can't increase the stack of a non-stackable modifier")

        self.amount += value
        if self.amount > self.max_amount:
            raise ValueError(f"Stacks of {self} can't be increased above {self.max_amount}")


class ModifierType(Enum):
    BUFF = 0
    DEBUFF = 1


class BaseModifier(ABC):
    def __init__(self, stack: Stack):
        self.stack = stack

    @property
    @abstractmethod
    def type(self) -> ModifierType:
        raise NotImplementedError

    @property
    def step(self) -> Optional[BaseStep]:
        return None

    def on_turn_end(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[ModifierEffect]:
        if self.stack.type.duration and self.stack.amount > self.stack.min_amount:
            self.stack.decrease(1)

        return []

    def on_turn_start(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[ModifierEffect]:
        return []

    def on_battle_end(
        self, owner: BaseActor, char: Character, monsters: MonsterGroup
    ) -> List[ModifierEffect]:
        return []
