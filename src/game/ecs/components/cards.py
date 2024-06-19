from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class CardCostComponent(BaseComponent):
    value: int


@dataclass
class CardRequiresTargetComponent(BaseComponent):
    pass


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
class CardIsActiveComponent(BaseComponent):
    pass


@dataclass
class CardStrikeComponent(BaseComponent):
    pass


@dataclass
class CardDefendComponent(BaseComponent):
    pass


@dataclass
class CardNeutralizeComponent(BaseComponent):
    pass


@dataclass
class CardSurvivorComponent(BaseComponent):
    pass
