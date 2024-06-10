from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent


class EffectSelectionType(Enum):
    NONE = "NONE"
    ALL = "ALL"
    RANDOM = "RANDOM"
    SPECIFIC = "SPECIFIC"


@dataclass
class EffectSelectionTypeComponent(BaseComponent):
    value: EffectSelectionType


@dataclass
class EffectQueryComponentsComponent(BaseComponent):
    value: list[type[BaseComponent]]


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


# TODO: improve effect lifecycle nomenclature
# TODO: remame priority to position
@dataclass
class EffectIsQueuedComponent(BaseComponent):
    priority: int


@dataclass
class EffectIsDispatchedComponent(BaseComponent):
    pass


@dataclass
class EffectIsPendingInputTargetsComponent(BaseComponent):
    pass


@dataclass
class EffectIsTargetedComponent(BaseComponent):
    pass


@dataclass
class EffectNumberOfTargetsComponent(BaseComponent):
    value: int


@dataclass
class EffectIsHaltedComponent(BaseComponent):
    priority: int
