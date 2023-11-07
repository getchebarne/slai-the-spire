from abc import ABC
from dataclasses import dataclass
from typing import Optional


MAX_BLOCK = 999


@dataclass
class Health:
    current: int
    max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max is None:
            self.max = self.current

    def __str__(self) -> str:
        return f"\u2764\uFE0F {self.current}/{self.max}"


@dataclass
class Block:
    current: int = 0
    max: int = MAX_BLOCK

    def __str__(self) -> str:
        return f"\U0001F6E1 {self.current}"


@dataclass
class Debuffs:
    vulnerable: int = 0
    weak: int = 0
    frail: int = 0


@dataclass
class Buffs:
    strength: int = 0
    dexterity: int = 0


class BaseActor(ABC):
    def __init__(
        self,
        health: Health,
        block: Block,
        buffs: Buffs,
        debuffs: Debuffs,
    ) -> None:
        self.health = health
        self.block = block
        self.buffs = buffs
        self.debuffs = debuffs

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.block} {self.health}"
