from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent


class EffectSelectionType(Enum):
    NONE = "NONE"
    ALL = "ALL"
    RANDOM = "RANDOM"


@dataclass
class EffectTargetComponent(BaseComponent):
    pass


@dataclass
class GainBlockEffectComponent(BaseComponent):
    value: int


@dataclass
class DealDamageEffectComponent(BaseComponent):
    value: int


@dataclass
class DiscardCardEffectComponent(BaseComponent):
    pass


@dataclass
class SetBlockToZeroEffect(BaseComponent):
    pass


@dataclass
class RefillEnergyEffect(BaseComponent):
    pass


@dataclass
class DiscardHandAtEndOfTurnEffect(BaseComponent):
    pass


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
class EffectToBeDispatchedComponent(BaseComponent):
    priority: int


@dataclass
class EffectIsDispatchedComponent(BaseComponent):
    pass
