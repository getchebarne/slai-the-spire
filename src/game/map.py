from dataclasses import dataclass, field
from enum import Enum


class RoomType(Enum):
    COMBAT_BOSS = "COMBAT_BOSS"
    COMBAT_ELITE = "COMBAT_ELITE"
    COMBAT_MONSTER = "COMBAT_MONSTER"
    QUESTION_MARK = "QUESTION_MARK"
    REST_SITE = "REST_SITE"
    SHOP = "SHOP"
    TREASURE = "TREASURE"


# TODO: add actual map generation algorithm
@dataclass(frozen=True)
class Map:
    room_types: list[RoomType] = field(default_factory=list)
