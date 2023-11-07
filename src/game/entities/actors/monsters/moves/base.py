from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import List

from game.effects.monster import MonsterEffect
from game.entities.actors.characters.base import Character


if TYPE_CHECKING:
    from game.entities.actors.monsters.base import Intent
    from game.entities.actors.monsters.base import Monster
    from game.entities.actors.monsters.group import MonsterGroup


class BaseMonsterMove(ABC):
    @property
    @abstractmethod
    def intent(self) -> Intent:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, owner: Monster, char: Character, monsters: MonsterGroup
    ) -> List[MonsterEffect]:
        raise NotImplementedError
