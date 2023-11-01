from abc import ABC
from enum import Enum
from typing import List

from game.effects.base import TargetType
from game.effects.card import CardEffect


# TODO: add card rarity
class CardType(Enum):
    ATTACK = 0
    SKILL = 1
    POWER = 2
    STATUS = 3
    CURSE = 4


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

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Determine if the card requires targetting based on its effects
        cls.requires_target = any(
            effect.target_type == TargetType.SINGLE for effect in cls.effects
        )
