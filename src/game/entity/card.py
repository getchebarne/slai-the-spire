from dataclasses import dataclass
from enum import Enum

from src.game.core.effect import Effect
from src.game.entity.base import EntityBase


class CardColor(Enum):
    COLORLESS = "COLORLESS"
    CURSE = "CURSE"
    GREEN = "GREEN"


class CardRarity(Enum):
    BASIC = "BASIC"
    COMMON = "COMMON"
    CURSE = "CURSE"
    RARE = "RARE"
    UNCOMMON = "UNCOMMON"
    SPECIAL = "SPECIAL"


class CardType(Enum):
    ATTACK = "ATTACK"
    CURSE = "CURSE"
    POWER = "POWER"
    SKILL = "SKILL"
    STATUS = "STATUS"


# TODO: add `upgraded` field
@dataclass
class EntityCard(EntityBase):
    name: str
    color: CardColor
    type: CardType
    cost: int
    rarity: CardRarity
    effects: list[Effect]
    exhaust: bool = False
    innate: bool = False
