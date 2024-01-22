from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from game.context import Entity


class EffectType(Enum):
    DAMAGE = 0
    BLOCK = 1
    WEAK = 2
    PLUS_STR = 3

    def __str__(self) -> str:
        return self.name


@dataclass
class Effect:
    # TODO: add created_by
    type: EffectType
    value: Optional[int]  # TODO: add default
    source: Entity
    target: Entity

    def __str__(self) -> str:
        return f"{self.type}: {self.value}"
