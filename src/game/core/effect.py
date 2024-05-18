from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.game.ecs.components import BaseComponent


class EffectType(Enum):
    DAMAGE = "DAMAGE"
    BLOCK = "BLOCK"
    WEAK = "WEAK"
    GAIN_STR = "GAIN_STR"
    DRAW_CARD = "DRAW_CARD"
    HEAL = "HEAL"


class SelectionType(Enum):
    NONE = "NONE"
    SPECIFIC = "SPECIFIC"
    RANDOM = "RANDOM"
    ALL = "ALL"


@dataclass
class Effect:
    # TODO: add created_by
    type: EffectType
    value: int
    query_components: list[type[BaseComponent]]
    selection_type: SelectionType

    def __str__(self) -> str:
        return f"{self.type}: {self.value}"
