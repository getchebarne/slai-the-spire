from dataclasses import dataclass
from typing import Optional

from game.entities.actors.base import BaseActor
from game.entities.actors.base import Block
from game.entities.actors.modifiers.group import ModifierGroup


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
        max_health: int,
        current_health: Optional[int] = None,
        block: Optional[Block] = None,
        modifiers: Optional[ModifierGroup] = None,
        energy: Optional[Energy] = None,
    ) -> None:
        super().__init__(max_health, current_health, block, modifiers)

        self.energy = energy if energy is not None else Energy()

    def __str__(self) -> str:
        base_str = super().__str__()
        return f"{base_str} {self.energy}"
