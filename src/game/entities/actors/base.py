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
    max: int
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max

        elif self.current > self.max:
            raise ValueError("Current health can't be larger than maximum health")

    def __str__(self) -> str:
        return f"\u2764\uFE0F {self.current}/{self.max}"


@dataclass
class Block:
    max: int = MAX_BLOCK
    current: int = 0

    def __str__(self) -> str:
        return f"\U0001F6E1 {self.current}"


class BaseActor(ABC):
    def __init__(
        self,
        health: Health,
        block: Optional[Block] = None,
        modifiers: Optional[ModifierGroup] = None,
    ) -> None:
        self.health = health
        self.block = block if block is not None else Block()
        self.modifiers = modifiers if modifiers is not None else ModifierGroup()

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
