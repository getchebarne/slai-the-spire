from dataclasses import dataclass, field
from enum import Enum
from typing import List

from game.core.effect import Effect


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

        self.amount = max(self.min_amount, self.amount - value)

    def increase(self, value: int) -> None:
        if self.type.none:
            raise ValueError("Can't increase the stack of a non-stackable modifier")

        self.amount = min(self.max_amount, self.amount + value)


class ModifierType(Enum):
    BUFF = 0
    DEBUFF = 1


@dataclass
class Modifier:
    name: str
    type: ModifierType
    stack: Stack
    battle_end_effects: List[Effect] = field(default_factory=list)
    turn_end_effects: List[Effect] = field(default_factory=list)
    turn_start_effects: List[Effect] = field(default_factory=list)
