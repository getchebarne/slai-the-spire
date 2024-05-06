from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EffectType(Enum):
    DAMAGE = 0
    BLOCK = 1
    WEAK = 2
    GAIN_STR = 3
    DRAW_CARD = 4
    HEAL = 5

    def __str__(self) -> str:
        return self.name


@dataclass
class Effect:
    # TODO: add created_by
    type: EffectType
    value: int
    source_entity_id: Optional[int] = None
    target_entity_id: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.type}: {self.value}"
