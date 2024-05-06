from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EffectType(Enum):
    DAMAGE = "DAMAGE"
    BLOCK = "BLOCK"
    WEAK = "WEAK"
    GAIN_STR = "GAIN_STR"
    DRAW_CARD = "DRAW_CARD"
    HEAL = "HEAL"


@dataclass
class Effect:
    # TODO: add created_by
    type: EffectType
    value: int
    source_entity_id: Optional[int] = None
    target_entity_id: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.type}: {self.value}"
