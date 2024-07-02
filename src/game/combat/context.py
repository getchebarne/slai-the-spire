from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class Entity(ABC):
    pass


@dataclass
class Health:
    max: int
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class Block:
    max: int = 999
    current: int = 0


@dataclass
class Actor(Entity):
    name: str
    health: Health
    block: Block = field(default_factory=Block)


@dataclass
class Character(Actor):
    pass


@dataclass
class Monster(Actor):
    pass


class EffectType(Enum):
    DEAL_DAMAGE = "DEAL_DAMAGE"
    GAIN_BLOCK = "GAIN_BLOCK"
    DRAW_CARD = "DRAW_CARD"
    REFILL_ENERGY = "REFILL_ENERGY"
    DISCARD = "DISCARD"
    ZERO_BLOCK = "ZERO_BLOCK"


class EffectTargetType(Enum):
    CHARACTER = "CHARACTER"
    MONSTER = "MONSTER"
    CARD_TARGET = "CARD_TARGET"
    CARD_IN_HAND = "CARD_IN_HAND"
    TURN = "TURN"


class EffectSelectionType(Enum):
    SPECIFIC = "SPECIFIC"
    ALL = "ALL"
    RANDOM = "RANDOM"


@dataclass
class Effect:
    type: EffectType
    value: Optional[int] = None
    target_type: Optional[EffectTargetType] = None
    selection_type: Optional[EffectSelectionType] = None


@dataclass
class Card(Entity):
    name: str
    cost: int
    effects: list[Effect]

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class Energy:
    max: int = 3
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class GameContext:
    # Specific entities
    character: Character
    monsters: list[Monster]
    energy: Energy
    deck: set[Card]
    hand: list[Card] = field(default_factory=list)
    draw_pile: list[Card] = field(default_factory=list)
    discard_pile: set[Card] = field(default_factory=set)

    # Active card
    active_card: Optional[Card] = None

    # Actor turn
    turn: Optional[Actor] = None

    # Effect queue
    effect_queue: deque[Effect] = field(default_factory=deque)

    # Effect processing
    effect_target: Optional[Entity] = None  # TODO: shrink type annotation?
    effect_value: Optional[int] = None
