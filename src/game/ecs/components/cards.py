from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class CardCostComponent(BaseComponent):
    value: int


@dataclass
class CardIsActiveComponent(BaseComponent):
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
class CardHasEffectsComponent(BaseComponent):
    effect_entity_ids: list[int]
