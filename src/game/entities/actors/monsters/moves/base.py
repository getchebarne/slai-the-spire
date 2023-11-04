from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from game.effects.monster import MonsterEffect
from game.entities.actors.characters.base import Character

if TYPE_CHECKING:
    from game.entities.actors.monsters.base import Intent
    from game.entities.actors.monsters.base import Monster
    from game.entities.actors.monsters.base import MonsterCollection


class BaseMonsterMove(ABC):
    @property
    @abstractmethod
    def intent(self) -> Intent:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, owner: Monster, char: Character, monsters: MonsterCollection
    ) -> List[MonsterEffect]:
        raise NotImplementedError
