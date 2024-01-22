from dataclasses import dataclass, field
from typing import List, Optional

from game.core.modifier import Modifier


MAX_BLOCK = 999


@dataclass
class Health:
    max: int
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max

        elif self.current > self.max:
            raise ValueError("Current health can't be larger than maximum health")

    def __str__(self) -> str:
        return f"\u2764\uFE0F {self.current}/{self.max}"


@dataclass
class Block:
    max: int = MAX_BLOCK
    current: int = 0

    def __str__(self) -> str:
        return f"\U0001F6E1 {self.current}"


@dataclass
class Actor:
    name: str
    base_health: int
    block: Block = field(default_factory=Block)
    modifiers: List[Modifier] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n{self.block} {self.health}\n{self.modifiers}"
