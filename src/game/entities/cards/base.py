from abc import ABC
from enum import Enum
from typing import List

from game.effects.card import CardEffect


# TODO: add card rarity
class CardType(Enum):
    ATTACK = 0
    SKILL = 1
    POWER = 2
    STATUS = 3
    CURSE = 4


# TODO: rename to `Card`
class BaseCard(ABC):
    name: str
    type_: CardType
    effects: List[CardEffect]

    def __init__(self, cost: int):
        self._cost = cost

    @property
    def cost(self) -> int:
        return self._cost

    @cost.setter
    def cost(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError(f"Cost must be an instance of {int}.")

        self._cost = value

    def __str__(self) -> str:
        return f"{type(self).__name__} ({self._cost})"