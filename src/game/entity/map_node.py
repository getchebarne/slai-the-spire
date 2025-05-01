from dataclasses import dataclass, field
from enum import Enum

from src.game.entity.base import EntityBase


class RoomType(Enum):
    COMBAT_BOSS = "COMBAT_BOSS"
    COMBAT_ELITE = "COMBAT_ELITE"
    COMBAT_MONSTER = "COMBAT_MONSTER"
    QUESTION_MARK = "QUESTION_MARK"
    REST_SITE = "REST_SITE"
    SHOP = "SHOP"
    TREASURE = "TREASURE"


@dataclass
class EntityMapNode(EntityBase):
    y: int
    x: int

    # Room type is initialized to `None` while the map layout is being created, filled after
    room_type: RoomType | None = None

    # Set storing the x-coordinates of connected nodes in the next level
    x_next: set[int] = field(default_factory=set)
