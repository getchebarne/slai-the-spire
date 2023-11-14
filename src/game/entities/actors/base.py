from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from game.entities.actors.modifiers.group import ModifierGroup


if TYPE_CHECKING:
    from game.effects.modifier import ModifierEffect
    from game.entities.actors.characters.base import Character
    from game.entities.actors.monsters.group import MonsterGroup

MAX_BLOCK = 999


@dataclass
class Health:
    current: int
    max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max is None:
            self.max = self.current

        elif self.current > self.max:
            raise ValueError("Current health can't be larger than maximum health")

    def __str__(self) -> str:
        return f"\u2764\uFE0F {self.current}/{self.max}"


@dataclass
class Block:
    current: int = 0
    max: int = MAX_BLOCK

    def __str__(self) -> str:
        return f"\U0001F6E1 {self.current}"


class BaseActor(ABC):
    def __init__(
        self, health: Health, block: Block = Block(), modifiers: ModifierGroup = ModifierGroup()
    ) -> None:
        self.health = health
        self.block = block
        self.modifiers = modifiers

    def on_turn_end(self, char: Character, monsters: MonsterGroup) -> List[ModifierEffect]:
        effects = []
        for modifier in self.modifiers:
            effects.extend(modifier.on_turn_end(self, char, monsters))

        return effects

    def on_turn_start(self, char: Character, monsters: MonsterGroup) -> List[ModifierEffect]:
        effects = []
        for modifier in self.modifiers:
            effects.extend(modifier.on_turn_start(self, char, monsters))

        return effects

    def on_battle_end(self, char: Character, monsters: MonsterGroup) -> List[ModifierEffect]:
        effects = []
        for modifier in self.modifiers:
            effects.extend(modifier.on_battle_end(self, char, monsters))

        return effects

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.block} {self.health} \n {self.modifiers}"
