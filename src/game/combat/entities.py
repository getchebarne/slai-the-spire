from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class Entity(ABC):
    pass


class EffectType(Enum):
    DEAL_DAMAGE = "DEAL_DAMAGE"
    GAIN_BLOCK = "GAIN_BLOCK"
    GAIN_WEAK = "GAIN_WEAK"
    DRAW_CARD = "DRAW_CARD"
    REFILL_ENERGY = "REFILL_ENERGY"
    DISCARD = "DISCARD"
    ZERO_BLOCK = "ZERO_BLOCK"
    PLAY_CARD = "PLAY_CARD"


class EffectTargetType(Enum):
    CHARACTER = "CHARACTER"
    MONSTER = "MONSTER"
    CARD_TARGET = "CARD_TARGET"
    CARD_IN_HAND = "CARD_IN_HAND"
    CARD_ACTIVE = "CARD_ACTIVE"
    TURN = "TURN"


class EffectSelectionType(Enum):
    INPUT = "INPUT"
    RANDOM = "RANDOM"


@dataclass
class Effect:
    type: EffectType
    value: Optional[int] = None
    target_type: Optional[EffectTargetType] = None
    selection_type: Optional[EffectSelectionType] = None


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
class ModifierType:
    WEAK = "WEAK"


@dataclass
class Modifier:
    stacks: Optional[int] = None
    stacks_min: Optional[int] = None
    stacks_max: Optional[int] = None
    stacks_duration: bool = False
    turn_start_effects: list[Effect] = field(default_factory=list)
    turn_end_effects: list[Effect] = field(default_factory=list)


@dataclass
class Actor(Entity):
    name: str
    health: Health
    block: Block = field(default_factory=Block)
    modifiers: dict[ModifierType, Modifier] = field(default_factory=dict)


@dataclass
class Character(Actor):
    pass


@dataclass
class MonsterMove:
    name: str
    effects: list[Effect]


@dataclass
class Monster(Actor):
    move: Optional[MonsterMove] = None


class CardName(Enum):
    STRIKE = "STRIKE"
    DEFEND = "DEFEND"
    NEUTRALIZE = "NEUTRALIZE"
    # SURVIVOR = "SURVIVOR"


@dataclass
class Card(Entity):
    name: str
    cost: int
    effects: list[Effect]

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False

        return id(self) == id(other)


@dataclass
class Energy:
    max: int = 3
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class Entities:
    # List mapping entity ids to entities
    entities: list[Entity] = field(default_factory=list)

    # Specific entities
    character_id: Optional[int] = None
    monster_ids: list[int] = field(default_factory=list)
    energy_id: Optional[int] = None
    card_in_deck_ids: set[int] = field(default_factory=set)
    card_in_draw_pile_ids: list[int] = field(default_factory=list)
    card_in_hand_ids: list[int] = field(default_factory=list)
    card_in_discard_pile_ids: set[int] = field(default_factory=set)

    # Card target
    card_target_id: Optional[int] = None

    # Active card
    card_active_id: Optional[int] = None

    # Effect target
    effect_target_id: Optional[list[int]] = None

    # TODO;
    entitiy_selectable_ids: Optional[list[int]] = None

    def create_entity(self, entity: Entity) -> int:
        entity_id = len(self.entities)
        self.entities.append(entity)

        return entity_id

    def get_entity(self, entity_id: int) -> Entity:
        return self.entities[entity_id]

    def get_character(self) -> Character:
        return self.entities[self.character_id]

    def get_monsters(self) -> list[Monster]:
        return [self.entities[monster_id] for monster_id in self.monster_ids]

    def get_energy(self) -> Energy:
        return self.entities[self.energy_id]

    def get_deck(self) -> set[Card]:
        return {self.entities[card_id] for card_id in self.card_in_deck_ids}

    def get_draw_pile(self) -> list[Card]:
        return [self.entities[card_id] for card_id in self.card_in_draw_pile_ids]

    def get_hand(self) -> list[Card]:
        return [self.entities[card_id] for card_id in self.card_in_hand_ids]

    def get_discard_pile(self) -> set[Card]:
        return {self.entities[card_id] for card_id in self.card_in_discard_pile_ids}

    def get_card_target(self) -> Optional[Entity]:
        if self.card_target_id is None:
            return None

        return self.entities[self.card_target_id]

    def get_active_card(self) -> Optional[Card]:
        if self.card_active_id is None:
            return None

        return self.entities[self.card_active_id]

    def get_actor_turn(self) -> Optional[Actor]:
        if self.actor_turn_id is None:
            return None

        return self.entities[self.actor_turn_id]

    def get_effect_target(self) -> Optional[Entity]:
        if self.effect_target_id is None:
            return None

        return self.entities[self.effect_target_id]
