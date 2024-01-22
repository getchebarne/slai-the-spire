from dataclasses import dataclass, field
from enum import Enum
from typing import List

from game.core.effect import Effect


class RelicRarity(Enum):
    STARTER = 0
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    SHOP = 4
    BOSS = 5
    EVENT = 6


@dataclass
class Relic:
    name: str
    rarity: RelicRarity
    battle_end_effects: List[Effect] = field(default_factory=list)
    battle_start_effects: List[Effect] = field(default_factory=list)
    turn_end_effects: List[Effect] = field(default_factory=list)
    turn_start_effects: List[Effect] = field(default_factory=list)
