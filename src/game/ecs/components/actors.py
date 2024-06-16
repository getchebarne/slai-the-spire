from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.base import BaseComponent
from src.game.ecs.components.base import BaseRelationshipComponent


MAX_BLOCK = 999


@dataclass
class HealthComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max


@dataclass
class BlockComponent(BaseComponent):
    max: int = MAX_BLOCK
    current: int = 0


@dataclass
class ActorComponent(BaseComponent):
    pass


@dataclass
class CharacterComponent(BaseComponent):
    pass


@dataclass
class MonsterComponent(BaseComponent):
    position: int


@dataclass
class IsTurnComponent(BaseComponent):
    pass


# TODO: move elsewhere
@dataclass
class DummyAIComponent(BaseComponent):
    pass


@dataclass
class MonsterPendingMoveUpdateComponent(BaseComponent):
    pass


@dataclass
class TurnStartComponent(BaseComponent):
    pass


@dataclass
class TurnEndComponent(BaseComponent):
    pass


@dataclass
class MonsterMoveComponent(BaseComponent):
    name: str


@dataclass
class MonsterMoveHasEffectsComponent(BaseRelationshipComponent):
    effect_entity_ids: list[int]


@dataclass
class MonsterHasMovesComponent(BaseRelationshipComponent):
    move_entity_ids: list[int]


@dataclass
class MonsterCurrentMoveComponent(BaseRelationshipComponent):
    pass


@dataclass
class MonsterMoveIntentComponent(BaseComponent):
    damage: int
    times: int
    block: bool


@dataclass
class ActorHasModifiersComponent(BaseRelationshipComponent):
    modifier_entity_ids: list[int]


@dataclass
class ModifierStacksComponent(BaseComponent):
    value: int


@dataclass
class ModifierMinimumStacksComponent(BaseComponent):
    value: int


@dataclass
class ModifierStacksDurationComponent(BaseComponent):
    pass


@dataclass
class ModifierWeakComponent(BaseComponent):
    pass


@dataclass
class IsEndingTurnComponent(BaseComponent):
    pass
