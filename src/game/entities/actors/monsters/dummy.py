from game.entities.actors.base import Block
from game.entities.actors.base import Buffs
from game.entities.actors.base import Debuffs
from game.entities.actors.base import Health
from game.entities.actors.monsters.base import Monster
from game.entities.actors.monsters.moves.attack import Attack
from game.entities.actors.monsters.moves.base import BaseMonsterMove
from game.entities.actors.monsters.moves.defend import Defend


# TODO: unshare
BASE_HEALTH = Health(10)
BASE_BLOCK = Block(0)
BASE_BUFFS = Buffs()
BASE_DEBUFFS = Debuffs()


move_attack = Attack(6)
move_defend = Defend(5)


class Dummy(Monster):
    moves = [Attack, Defend]

    def __init__(
        self,
        health: Health = BASE_HEALTH,
        block: Block = BASE_BLOCK,
        buffs: Buffs = BASE_BUFFS,
        debuffs: Debuffs = BASE_DEBUFFS,
    ) -> None:
        super().__init__(health, block, buffs, debuffs)

    def update_move(self) -> None:
        if isinstance(self.move, Attack):
            self.move = move_defend
        elif isinstance(self.move, Defend):
            self.move = move_attack
        else:
            raise ValueError(f"Unexpected move: {self.move}")

    def _first_move(self) -> BaseMonsterMove:
        return move_attack
