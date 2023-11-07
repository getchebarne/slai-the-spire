from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import List

from game.effects.monster import MonsterEffect
from game.entities.actors.base import BaseActor
from game.entities.actors.base import Block
from game.entities.actors.base import Buffs
from game.entities.actors.base import Debuffs
from game.entities.actors.base import Health
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.moves.base import BaseMonsterMove

if TYPE_CHECKING:
    from game.entities.actors.monsters.group import MonsterGroup


# TODO: probably set defaults to `None`
# TODO: apply intent correction based on buffs / debuffs (e.g., weak)?
@dataclass
class Intent:
    damage: int = 0
    instances: int = 0
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
        if self.damage:
            str_ = f"{str_}\U0001F5E1 {self.damage}"

        if self.instances > 1:
            str_ = f"{str_}x{self.instances}"

        return str_


class Monster(BaseActor):
    def __init__(
        self,
        health: Health,
        block: Block,
        buffs: Buffs,
        debuffs: Debuffs,
    ) -> None:
        super().__init__(health, block, buffs, debuffs)

        # Check if `moves` is defined
        if not hasattr(self.__class__, "moves") or not self.__class__.moves:
            raise NotImplementedError(
                "Subclasses of `Monster` must define a `moves` class variable"
            )

        # Set initial move
        self.move = self._first_move()

    def execute_move(self, char: Character, monsters: MonsterGroup) -> List[MonsterEffect]:
        return self.move(self, char, monsters)

    @abstractmethod
    def update_move(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _first_move(self) -> BaseMonsterMove:
        raise NotImplementedError

    def __str__(self) -> str:
        # TODO: improve intent
        # Get BaseActor string
        base_str = super().__str__()

        # Append intent
        return f"{base_str} {self.move.intent}"
