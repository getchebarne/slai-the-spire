from dataclasses import dataclass, field
from enum import Enum

from src.game.core.effect import Effect
from src.game.entity.actor import EntityActor


@dataclass(frozen=True)
class Intent:
    damage: int | None = None
    instances: int | None = None
    block: bool = False
    buff: bool = False
    debuff_powerful: bool = False


class MonsterType(Enum):
    BOSS = "BOSS"
    ELITE = "ELITE"
    NORMAL = "NORMAL"


@dataclass(frozen=True)
class MonsterMove:
    effects: list[Effect]
    intent: Intent


@dataclass
class EntityMonster(EntityActor):
    type: MonsterType = MonsterType.NORMAL
    moves: dict[str, MonsterMove] = field(default_factory=dict)
    move_name_current: str | None = None
    move_name_history: list[str] = field(default_factory=list)
