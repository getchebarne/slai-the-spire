from dataclasses import dataclass

from game.entities.actors.base import BaseActor
from game.entities.actors.base import Block
from game.entities.actors.base import Buffs
from game.entities.actors.base import Debuffs
from game.entities.actors.base import Health


BASE_ENERGY = 3


@dataclass
class Energy:
    current: int = BASE_ENERGY
    max: int = BASE_ENERGY

    def __str__(self) -> str:
        return f"\U0001F50B {self.current}/{self.max}"


class Character(BaseActor):
    def __init__(
        self,
        health: Health,
        block: Block = Block(),
        buffs: Buffs = Buffs(),
        debuffs: Debuffs = Debuffs(),
        energy: Energy = Energy(),
    ) -> None:
        super().__init__(health, block, buffs, debuffs)

        self.energy = energy

    def __str__(self) -> str:
        base_str = super().__str__()
        return f"{base_str} {self.energy}"