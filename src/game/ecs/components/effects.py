from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent
from src.game.ecs.components.base import BaseRelationshipComponent
from src.game.ecs.components.base import BaseSingletonComponent


class EffectSelectionType(Enum):
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


@dataclass
class EffectGainBlockComponent(BaseComponent):
    value: int


@dataclass
class EffectDealDamageComponent(BaseComponent):
    value: int


@dataclass
class EffectDiscardCardComponent(BaseComponent):
    pass


@dataclass
class EffectSetBlockToZero(BaseComponent):
    pass


@dataclass
class EffectRefillEnergyComponent(BaseComponent):
    pass


@dataclass
class EffectDrawCardComponent(BaseComponent):
    value: int


@dataclass
class EffectShuffleDiscardPileIntoDrawPileComponent(BaseComponent):
    pass


@dataclass
class EffectShuffleDeckIntoDrawPileComponent(BaseComponent):
    pass


@dataclass
class EffectIsQueuedComponent(BaseComponent):
    position: int


@dataclass
class EffectIsDispatchedComponent(BaseComponent):
    pass


# TODO: rename
@dataclass
class EffectTurnEndComponent(BaseComponent):
    pass


@dataclass
class EffectTurnStartComponent(BaseComponent):
    pass


@dataclass
class EffectIsPendingInputTargetsComponent(BaseComponent):
    pass


@dataclass
class EffectIsTargetedSingletonComponent(BaseSingletonComponent):
    pass


@dataclass
class EffectNumberOfTargetsComponent(BaseComponent):
    value: int


@dataclass
class EffectIsHaltedComponent(BaseComponent):
    position: int


@dataclass
class EffectInputTargetComponent(BaseComponent):
    pass


@dataclass
class EffectModifierDeltaComponent(BaseComponent):
    value: int


@dataclass
class EffectCreateWeakComponent(BaseComponent):
    pass


@dataclass
class EffectParentComponent(BaseRelationshipComponent):
    entity_id: int
