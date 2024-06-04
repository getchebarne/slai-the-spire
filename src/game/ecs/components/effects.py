from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent


class EffectSelectionType(Enum):
    NONE = "NONE"
    ALL = "ALL"
    RANDOM = "RANDOM"
    SPECIFIC = "SPECIFIC"


@dataclass
class EffectTargetComponent(BaseComponent):
    pass


# TODO: swap name order (e.g., EffectGainBlockComponent)
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


# TODO: consolidate effect targeting data
@dataclass
class EffectSelectionTypeComponent(BaseComponent):
    value: EffectSelectionType


@dataclass
class EffectQueryComponentsComponent(BaseComponent):
    value: list[type[BaseComponent]]


# TODO: improve effect lifecycle nomeclature
@dataclass
class EffectToBeDispatchedComponent(BaseComponent):
    priority: int


@dataclass
class EffectIsDispatchedComponent(BaseComponent):
    pass


@dataclass
class EffectIsTargetedComponent(BaseComponent):
    pass
