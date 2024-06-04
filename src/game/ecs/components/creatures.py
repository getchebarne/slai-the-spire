from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.base import BaseComponent


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
class CharacterComponent(BaseComponent):
    pass


@dataclass
class MonsterComponent(BaseComponent):
    position: int


@dataclass
class MonsterMoveComponent(BaseComponent):
    name: str
    effect_entity_ids: list[int]


@dataclass
class MonsterIsTurnComponent(BaseComponent):
    pass


@dataclass
class MonsterPendingTurnComponent(BaseComponent):
    pass


@dataclass
class IsTurnComponent(BaseComponent):
    pass
