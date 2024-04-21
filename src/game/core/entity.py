from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from src.game.core.modifier import Modifier


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


@dataclass
class Block:
    max: int = MAX_BLOCK
    current: int = 0

    def __post_init__(self) -> None:
        if self.current > self.max:
            raise ValueError("Current block can't be larger than maximum block")


@dataclass
class Entity(ABC):
    name: str
    health: Health
    block: Block = field(default_factory=Block)
    modifiers: list[Modifier] = field(default_factory=list)
