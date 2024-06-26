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
class TurnComponent(BaseComponent):
    pass


@dataclass
class NextTurnComponent(BaseComponent):
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


class MonsterMoveIsQueuedComponent(BaseComponent):
    pass


@dataclass
class MonsterMoveDummyAttackComponent(BaseComponent):
    pass


@dataclass
class MonsterMoveDummyDefendComponent(BaseComponent):
    pass


@dataclass
class MonsterIntentDamageComponent(BaseComponent):
    value: int


@dataclass
class MonsterIntentTimesComponent(BaseComponent):
    value: int


@dataclass
class MonsterIntentBlockComponent(BaseComponent):
    pass


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
class ModifierParentComponent(BaseRelationshipComponent):
    actor_entity_id: int
