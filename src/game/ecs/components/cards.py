from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class CardCostComponent(BaseComponent):
    value: int


@dataclass
class ActiveCardComponent(BaseComponent):
    pass


@dataclass
class CardInDeckComponent(BaseComponent):
    pass


@dataclass
class CardInHandComponent(BaseComponent):
    position: int


@dataclass
class CardInDrawPileComponent(BaseComponent):
    position: int


@dataclass
class CardInDiscardPileComponent(BaseComponent):
    pass
