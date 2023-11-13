from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List

from game.effects.monster import MonsterEffect
from game.entities.actors.base import BaseActor
from game.entities.actors.base import Block
from game.entities.actors.base import Health
from game.entities.actors.characters.base import Character
from game.entities.actors.modifiers.group import ModifierGroup
from game.entities.actors.monsters.moves.base import BaseMonsterMove


if TYPE_CHECKING:
    from game.entities.actors.monsters.group import MonsterGroup


class Monster(BaseActor):
    def __init__(
        self, health: Health, block: Block = Block(), modifiers: ModifierGroup = ModifierGroup()
    ) -> None:
        super().__init__(health, block, modifiers)

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
        # Get BaseActor string
        base_str = super().__str__()

        # Append intent
        return f"{base_str} {self.move.intent}"
