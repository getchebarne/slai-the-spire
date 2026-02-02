from dataclasses import dataclass, field
from enum import Enum

from src.game.entity.base import EntityBase


class ModifierType(Enum):
    ACCURACY = "ACCURACY"
    AFTER_IMAGE = "AFTER_IMAGE"
    BLUR = "BLUR"
    BURST = "BURST"
    # CURL_UP = "CURL_UP"
    DEXTERITY = "DEXTERITY"
    DOUBLE_DAMAGE = "DOUBLE_DAMAGE"
    INFINITE_BLADES = "INFINITE_BLADES"
    MODE_SHIFT = "MODE_SHIFT"
    NEXT_TURN_BLOCK = "NEXT_TURN_BLOCK"
    NEXT_TURN_ENERGY = "NEXT_TURN_ENERGY"
    PHANTASMAL = "PHANTASMAL"
    RITUAL = "RITUAL"
    SHARP_HIDE = "SHARP_HIDE"
    SPORE_CLOUD = "SPORE_CLOUD"
    STRENGTH = "STRENGTH"
    THOUSAND_CUTS = "THOUSAND_CUTS"
    VULNERABLE = "VULNERABLE"
    WEAK = "WEAK"


@dataclass(frozen=True)
class ModifierConfig:
    is_buff: bool
    stacks_duration: bool
    stacks_min: int = 1
    stacks_max: int = 999


@dataclass
class ModifierData:
    config: ModifierConfig
    is_new: bool
    stacks_current: int


@dataclass
class EntityActor(EntityBase):
    name: str
    health_current: int
    health_max: int
    block_current: int = 0
    modifier_map: dict[ModifierType, ModifierData] = field(default_factory=dict)
