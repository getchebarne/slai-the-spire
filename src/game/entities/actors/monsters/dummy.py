from typing import Optional

from game.entities.actors.base import Block
from game.entities.actors.modifiers.group import ModifierGroup
from game.entities.actors.monsters.base import Monster
from game.entities.actors.monsters.moves.attack import Attack
from game.entities.actors.monsters.moves.base import BaseMonsterMove
from game.entities.actors.monsters.moves.defend import Defend


BASE_HEALTH = 10


move_attack = Attack(6)
move_defend = Defend(5)


class Dummy(Monster):
    moves = [Attack, Defend]

    def __init__(
        self,
        max_health: int = BASE_HEALTH,
        current_health: Optional[int] = None,
        block: Optional[Block] = None,
        modifiers: Optional[ModifierGroup] = None,
    ) -> None:
        super().__init__(max_health, current_health, block, modifiers)

    def update_move(self) -> None:
        if isinstance(self.move, Attack):
            self.move = move_defend

        elif isinstance(self.move, Defend):
            self.move = move_attack

        else:
            raise ValueError(f"Unexpected move: {self.move}")

    def _first_move(self) -> BaseMonsterMove:
        return move_attack
