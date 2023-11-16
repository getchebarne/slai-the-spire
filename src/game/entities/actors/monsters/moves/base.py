from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from game.effects.monster import MonsterEffect
from game.entities.actors.characters.base import Character


if TYPE_CHECKING:
    from game.entities.actors.monsters.base import Monster
    from game.entities.actors.monsters.group import MonsterGroup


# TODO: apply intent correction based on buffs / debuffs (e.g., weak)?
@dataclass
class Intent:
    damage: Optional[int] = None
    instances: Optional[int] = None
    block: bool = False
    buff: bool = False
    debuff: bool = False
    strong_debuff: bool = False
    escape: bool = False
    asleep: bool = False
    stunned: bool = False
    unknown: bool = False

    def __str__(self) -> str:
        # TODO: add support for other intents
        str_ = ""
        if self.damage is not None:
            str_ = f"{str_}\U0001F5E1 {self.damage}"

        if self.instances is not None and self.instances > 1:
            str_ = f"{str_}x{self.instances}"

        if self.block:
            str_ = f"{str_} \U0001F6E1"

        return str_


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
