from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Optional


MAX_BLOCK = 999


@dataclass
class BaseComponent(ABC):
    pass


# COMMON
################


@dataclass
class NameComponent(BaseComponent):
    value: str


# CREATURE
################


@dataclass
class HealthComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max


@dataclass
class BlockComponent(BaseComponent):
    max: int = MAX_BLOCK
    current: int = 0


@dataclass
class CharacterComponent(BaseComponent):
    pass


@dataclass
class MonsterComponent(BaseComponent):
    pass


# ENERGY
################


@dataclass
class EnergyComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max


# TARGET
################


@dataclass
class TargetComponent(BaseComponent):
    pass


# EFFECT
################


class SelectionType(Enum):
    NONE = "NONE"
    SPECIFIC = "SPECIFIC"
    RANDOM = "RANDOM"
    ALL = "ALL"


@dataclass
class GainBlockEffectComponent(BaseComponent):
    value: int


@dataclass
class DealDamageEffectComponent(BaseComponent):
    value: int


@dataclass
class EffectSelectionTypeComponent(BaseComponent):
    value: SelectionType


@dataclass
class EffectQueryComponentsComponent(BaseComponent):
    value: list[type[BaseComponent]]


@dataclass
class HasEffectsComponent(BaseComponent):
    entity_ids: list[int]  # TODO: rename?


@dataclass
class EffectToBeTargetedComponent(BaseComponent):
    pass


@dataclass
class EffectApplyToComponent(BaseComponent):
    entity_ids: int


# CARD
################


@dataclass
class CardComponent(BaseComponent):
    pass


@dataclass
class ActiveCardComponent(BaseComponent):
    pass


@dataclass
class CardInHandComponent(BaseComponent):
    position: int


@dataclass
class CardCostComponent(BaseComponent):
    cost: int
