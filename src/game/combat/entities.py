from dataclasses import dataclass, field

from src.game.combat.effect import Effect


@dataclass
class Entity:
    pass


@dataclass
class Modifier:
    stacks_current: int | None = None
    stacks_min: int | None = None
    stacks_max: int | None = None
    stacks_duration: bool = False
    turn_start_effects: list[Effect] = field(default_factory=list)
    turn_end_effects: list[Effect] = field(default_factory=list)


@dataclass
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


@dataclass
class Character(Actor):
    pass


@dataclass
class MonsterMove:
    name: str
    effects: list[Effect]


@dataclass
class Monster(Actor):
    move_current: MonsterMove | None = None
    move_history: list[MonsterMove] = field(default_factory=list)


@dataclass
class Card(Entity):
    name: str
    cost: int
    effects: list[Effect]


@dataclass
class Energy(Entity):
    max: int = 3
    current: int | None = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class EntityManager:
    entities: list[Entity]

    id_character: int | None = None
    id_monsters: list[int] | None = None
    id_energy: int | None = None
    id_cards_in_deck: list[int] | None = None
    id_cards_in_draw_pile: list[int] = field(default_factory=list)
    id_cards_in_hand: list[int] = field(default_factory=list)
    id_cards_in_disc_pile: list[int] = field(default_factory=list)
    id_card_active: int | None = None

    # TODO: can maybe fuse into one single `id_target`?
    id_card_target: int | None = None
    id_effect_target: int | None = None

    id_selectables: list[int] | None = None


def create_entity(entity_manager: EntityManager, entitiy: Entity) -> int:
    entity_manager.entities.append(entitiy)

    # Return the entity's index
    return len(entity_manager.entities) - 1
