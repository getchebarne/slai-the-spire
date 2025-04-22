from dataclasses import dataclass
from enum import Enum


class RoomType(Enum):
    COMBAT = "COMBAT"
    SHOP = "SHOP"
    REST_SITE = "REST_SITE"
    EVENT = "EVENT"
    TREASURE = "TREASURE"


@dataclass(frozen=True)
class Room:
    name: str
    type: RoomType
