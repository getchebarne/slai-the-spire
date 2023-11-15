from dataclasses import dataclass, field, fields
from typing import Generator

from game.entities.actors.modifiers.base import BaseModifier
from game.entities.actors.modifiers.strength import Strength
from game.entities.actors.modifiers.weak import Weak


@dataclass
class ModifierGroup:
    strength: Strength = field(default_factory=Strength)
    # dexterity: int = 0
    # vulnerable: int = 0
    weak: Weak = field(default_factory=Weak)
    # frail: int = 0

    def __iter__(self) -> Generator[BaseModifier, None, None]:
        for field_ in fields(self):
            yield getattr(self, field_.name)

    def __str__(self) -> str:
        return f"\U0001F5E1 {self.strength.stack.amount} \U0001F940 {self.weak.stack.amount}"
