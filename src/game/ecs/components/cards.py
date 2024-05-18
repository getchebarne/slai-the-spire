from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class CardComponent(BaseComponent):
    pass


@dataclass
class CardCostComponent(BaseComponent):
    value: int


@dataclass
class ActiveCardComponent(BaseComponent):
    pass


@dataclass
class CardInHandComponent(BaseComponent):
    position: int
