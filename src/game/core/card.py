from dataclasses import dataclass
from enum import Enum

from game.logic.card.base import BaseCardLogic


# TODO: add card upgrade


class CardRarity(Enum):
    BASIC = 0
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    SPECIAL = 4


class CardType(Enum):
    ATTACK = 0
    SKILL = 1
    POWER = 2
    STATUS = 3
    CURSE = 4


@dataclass
class Card:
    name: str
    desc: str
    cost: int
    type: CardType
    rarity: CardRarity
    logic: BaseCardLogic

    def __str__(self) -> str:
        return f"{self.name} ({self.cost})"
