from dataclasses import dataclass, field
from enum import Enum

from src.game.entity.base import EntityBase


class ModifierType(Enum):
    WEAK = "WEAK"
    STRENGTH = "STRENGTH"
    SPORE_CLOUD = "SPORE_CLOUD"
    VULNERABLE = "VULNERABLE"


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
