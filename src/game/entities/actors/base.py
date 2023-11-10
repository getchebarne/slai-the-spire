from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from game.entities.actors.modifiers.strength import Strength


MAX_BLOCK = 999


@dataclass
class Health:
    current: int
    max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max is None:
            self.max = self.current

        elif self.current > self.max:
            raise ValueError("Current health can't be larger than maximum health")

    def __str__(self) -> str:
        return f"\u2764\uFE0F {self.current}/{self.max}"


@dataclass
class Block:
    current: int = 0
    max: int = MAX_BLOCK

    def __str__(self) -> str:
        return f"\U0001F6E1 {self.current}"


@dataclass
class Modifiers:
    strength: Strength = Strength()
    dexterity: int = 0
    vulnerable: int = 0
    weak: int = 0
    frail: int = 0


class BaseActor(ABC):
    def __init__(
        self, health: Health, block: Block = Block(), modifiers: Modifiers = Modifiers()
    ) -> None:
        self.health = health
        self.block = block
        self.modifiers = modifiers

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.block} {self.health}"
