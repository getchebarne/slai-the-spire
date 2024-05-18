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
    pass


@dataclass
class MonsterMoveComponent(BaseComponent):
    move: Optional[str] = None  # TODO: change default
