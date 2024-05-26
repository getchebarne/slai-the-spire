from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent


MAX_BLOCK = 999


class EffectSelectionType(Enum):
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
class DiscardEffectComponent(BaseComponent):
    value: int


@dataclass
class GainWeakEffectComponent(BaseComponent):
    value: int


@dataclass
class DrawCardEffectComponent(BaseComponent):
    value: int


@dataclass
class ShuffleDiscardPileIntoDrawPileEffectComponent(BaseComponent):
    pass


@dataclass
class ShuffleDeckIntoDrawPileEffectComponent(BaseComponent):
    pass


@dataclass
class EffectSelectionTypeComponent(BaseComponent):
    value: EffectSelectionType


@dataclass
class EffectQueryComponentsComponent(BaseComponent):
    value: list[type[BaseComponent]]


@dataclass
class HasEffectsComponent(BaseComponent):
    entity_ids: list[int]  # TODO: rename?


@dataclass
class EffectToBeTargetedComponent(BaseComponent):
    priority: int


@dataclass
class EffectApplyToComponent(BaseComponent):
    entity_ids: list[int]
