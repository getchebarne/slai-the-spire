from abc import ABC
from abc import abstractmethod
from enum import Enum
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional

from game.effects.card import CardEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup


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

    @abstractmethod
    def use(
        self,
        char: Character,
        monsters: MonsterGroup,
        target_monster_idx: Optional[int] = None,
    ) -> List[CardEffect]:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{type(self).__name__} ({self._cost})"


def ensure_target_monster_idx(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self: BaseCard,
        char: Character,
        monsters: MonsterGroup,
        target_monster_idx: Optional[int] = None,
    ) -> List[CardEffect]:
        if target_monster_idx is None:
            raise ValueError("target_monster_idx cannot be None for this card")

        return func(self, char, monsters, target_monster_idx)

    return wrapper
