from abc import ABC
from dataclasses import dataclass
from typing import Optional

from src.game.core.effect import Effect


@dataclass
class BaseComponent(ABC):
    pass


@dataclass
class HealthComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class BlockComponent(BaseComponent):
    current: int = 0


@dataclass
class EnergyComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


##############
#   TARGET   #
##############
@dataclass
class TargetComponent(BaseComponent):
    pass


##############
#   EFFECT   #
##############
@dataclass
class EffectsOnUseComponent(BaseComponent):
    effects: list[Effect]


@dataclass
class EffectsToBeTargetedComponent(BaseComponent):
    effects: list[Effect]


@dataclass
class EffectsToBeAppliedComponent(BaseComponent):
    effects: list[Effect]


############
#   CARD   #
############


@dataclass
class ActiveCardComponent(BaseComponent):
    pass


@dataclass
class CardInHandComponent(BaseComponent):
    position: int


@dataclass
class NameComponent(BaseComponent):
    value: str


@dataclass
class MonsterComponent(BaseComponent):
    pass


@dataclass
class MonsterMoveComponent(BaseComponent):
    move: Optional[str] = None  # TODO: change default


@dataclass
class CharacterComponent(BaseComponent):
    pass


@dataclass
class CardComponent(BaseComponent):
    pass


@dataclass
class CardCostComponent(BaseComponent):
    cost: int


@dataclass
class RelicComponent(BaseComponent):
    pass
