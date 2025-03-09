from dataclasses import dataclass, field, replace
from enum import Enum


# TODO: split this into multiple scripts


@dataclass(frozen=True)
class Entity:
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
    MOD_TICK = "MOD_TICK"  # TODO: rename
    GAIN_STR = "GAIN_STR"
    DECREASE_ENERGY = "DECREASE_ENERGY"


class EffectTargetType(Enum):
    CHARACTER = "CHARACTER"
    MONSTER = "MONSTER"
    CARD_TARGET = "CARD_TARGET"
    CARD_IN_HAND = "CARD_IN_HAND"
    CARD_ACTIVE = "CARD_ACTIVE"
    SOURCE = "SOURCE"


class EffectSelectionType(Enum):
    INPUT = "INPUT"
    RANDOM = "RANDOM"


@dataclass(frozen=True)
class Effect:
    type: EffectType
    value: int | None = None
    target_type: EffectTargetType | None = None
    selection_type: EffectSelectionType | None = None


@dataclass(frozen=True)
class Modifier:
    stacks_current: int | None = None
    stacks_min: int | None = None
    stacks_max: int | None = None
    stacks_duration: bool = False
    turn_start_effects: list[Effect] = field(default_factory=list)
    turn_end_effects: list[Effect] = field(default_factory=list)


@dataclass(frozen=True)
class Actor(Entity):
    name: str
    health_current: int
    health_max: int
    block_current: int = 0

    # Modifiers
    modifier_weak: Modifier = field(
        default_factory=lambda: Modifier(0, stacks_min=0, stacks_max=999, stacks_duration=True)
    )
    modifier_strength: Modifier = field(
        default_factory=lambda: Modifier(0, stacks_min=0, stacks_max=999, stacks_duration=False)
    )


@dataclass(frozen=True)
class Character(Actor):
    pass


@dataclass(frozen=True)
class MonsterMove:
    name: str
    effects: list[Effect]


@dataclass(frozen=True)
class Monster(Actor):
    move_current: MonsterMove | None = None  # TODO: revisit None
    move_history: list[MonsterMove] = field(default_factory=list)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Energy(Entity):
    max: int = 3
    current: int | None = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


# TODO: rename?
@dataclass(frozen=True)
class Entities:
    all: list[Entity] = field(default_factory=list)

    # Specific entities
    character_id: int | None = None
    monster_ids: list[int] = field(default_factory=list)
    energy_id: int | None = None
    card_in_deck_ids: set[int] = field(default_factory=set)
    card_in_draw_pile_ids: list[int] = field(default_factory=list)
    card_in_hand_ids: list[int] = field(default_factory=list)
    card_in_discard_pile_ids: set[int] = field(default_factory=set)

    # Card target # TODO: should merge w/ effect_target_id
    card_target_id: int | None = None

    # Active card
    card_active_id: int | None = None

    # Effect target
    effect_target_id: int | None = None

    # TODO;
    entity_selectable_ids: list[int] = field(default_factory=list)

    #
    actor_turn_id: int | None = None

    # TODO: merge?
    # effect_source_id: int | None = None
    # effect_target_id: int | None = None


def get_entity(entities: Entities, entity_id: int) -> Entity:
    return entities.all[entity_id]  # TODO: `Entities` could be just a list


# TODO: add single-entity counterpart?
def add_entities(entities: Entities, *new_entities: Entity) -> tuple[Entities, list[int]]:
    # Create a copy of the current entities list
    new_entities_list = list(entities.all)

    # Track the IDs assigned to the new entities
    new_entity_ids = []

    # Add each entity and record its ID
    for entity in new_entities:
        entity_id = len(new_entities_list)
        new_entities_list.append(entity)
        new_entity_ids.append(entity_id)

    # Create and return a new Entities instance with the updated list
    return replace(entities, all=new_entities_list), new_entity_ids
