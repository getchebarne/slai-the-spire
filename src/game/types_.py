from enum import Enum
from typing import TypeAlias


CardUpgraded: TypeAlias = bool
AscensionLevel: TypeAlias = int
Floor: TypeAlias = int


class RoomType(Enum):
    COMBAT_BOSS = "COMBAT_BOSS"
    COMBAT_ELITE = "COMBAT_ELITE"
    COMBAT_MONSTER = "COMBAT_MONSTER"
    QUESTION_MARK = "QUESTION_MARK"
    REST_SITE = "REST_SITE"
    SHOP = "SHOP"
    TREASURE = "TREASURE"


MapNodeXY: TypeAlias = tuple[int, int]  # (x, y)
MapNodes: TypeAlias = list[list[RoomType | None]]
MapEdges: TypeAlias = list[tuple[MapNodeXY, MapNodeXY]]  # (source, target)


class CombatResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
