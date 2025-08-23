from dataclasses import dataclass, field
from enum import Enum

from src.game.entity.base import EntityBase


class ModifierType(Enum):
    ACCURACY = "ACCURACY"
    MODE_SHIFT = "MODE_SHIFT"
    NEXT_TURN_BLOCK = "NEXT_TURN_BLOCK"
    NEXT_TURN_ENERGY = "NEXT_TURN_ENERGY"
    RITUAL = "RITUAL"
    SHARP_HIDE = "SHARP_HIDE"
    SPORE_CLOUD = "SPORE_CLOUD"
    STRENGTH = "STRENGTH"
    VULNERABLE = "VULNERABLE"
    WEAK = "WEAK"


@dataclass
class ModifierData:
    stacks_current: int | None = None
    stacks_min: int | None = None
    stacks_max: int | None = None
    stacks_duration: bool = False


@dataclass
class EntityActor(EntityBase):
    name: str
    health_current: int
    health_max: int
    block_current: int = 0
    modifier_map: dict[ModifierType, ModifierData] = field(default_factory=dict)
