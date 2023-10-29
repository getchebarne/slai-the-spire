from game.effects.base import TargetType
from game.effects.monster import MonsterEffect
from game.entities.actors.base import Block
from game.entities.actors.base import Buffs
from game.entities.actors.base import Debuffs
from game.entities.actors.base import Health
from game.entities.actors.monster import Monster


# TODO: unshare
BASE_HEALTH = Health(10)
BASE_BLOCK = Block(0)
BASE_BUFFS = Buffs()
BASE_DEBUFFS = Debuffs()


class Dummy(Monster):
    moves = {
        "attack": [MonsterEffect(target_type=TargetType.SINGLE, damage=6)],
        "defend": [MonsterEffect(target_type=TargetType.SELF, block=5)],
    }

    def __init__(
        self,
        health: Health = BASE_HEALTH,
        block: Block = BASE_BLOCK,
        buffs: Buffs = BASE_BUFFS,
        debuffs: Debuffs = BASE_DEBUFFS,
    ) -> None:
        super().__init__(health, block, buffs, debuffs)

    def update_move(self) -> None:
        if self.move == self.moves["attack"]:
            self.move = self.moves["defend"]
        else:
            self.move = self.moves["attack"]

        self.intent = self._move_to_intent(self.move)

    def _set_first_move(self) -> None:
        self.move = self.moves["attack"]
        self.intent = self._move_to_intent(self.move)
