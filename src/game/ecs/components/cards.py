from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent
from src.game.ecs.components.base import BaseRelationshipComponent


@dataclass
class CardCostComponent(BaseComponent):
    value: int


@dataclass
class CardIsPlayedComponent(BaseComponent):
    pass


@dataclass
class CardTargetComponent(BaseComponent):
    pass


@dataclass
class CardInDeckComponent(BaseComponent):
    pass


@dataclass
class CardInDrawPileComponent(BaseComponent):
    position: int


@dataclass
class CardInHandComponent(BaseComponent):
    position: int


@dataclass
class CardInDiscardPileComponent(BaseComponent):
    pass


@dataclass
class CardHasEffectsComponent(BaseRelationshipComponent):
    effect_entity_ids: list[int]


@dataclass
class CardIsActiveComponent(BaseComponent):
    pass
