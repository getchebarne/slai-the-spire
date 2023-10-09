from __future__ import annotations

from abc import ABC
from typing import Dict

MAX_BLOCK = 999
START_STATUS = {"weak": 0, "vulnerable": 0, "frail": 0}


class BaseEntity(ABC):
    def __init__(
        self,
        health: int,
        max_health: int,
        block: int = 0,
        status: Dict[str, int] = START_STATUS,
    ) -> None:
        self.health = health
        self.max_health = max_health
        self.block = block
        self.status = status

    def reset_block(self) -> None:
        self.block = 0

    def gain_block(self, block: int) -> None:
        self.block = min(MAX_BLOCK, self.block + block)

    def gain_health(self, health: int) -> None:
        self.health = min(self.max_health, self.health + health)

    def lose_health(self, health: int) -> None:
        self.health = max(0, self.health - health)

    def take_damage(self, damage: int) -> None:
        damage_over_block = max(0, damage - self.block)
        self.lose_health(damage_over_block)
        self.block = max(0, self.block - damage)

    def deal_damage(self, damage: int, target: BaseEntity) -> None:
        # TODO: improve this
        if self.status["weak"]:
            damage = int(damage * 0.75)

        target.take_damage(damage)

    def inc_status(self, name: str, duration: int) -> None:
        self.status[name] += duration

    def dec_status(self, name: str, duration: int = 1) -> None:
        self.status[name] = max(self.status[name] - duration, 0)

    def end_turn(self) -> None:
        # TODO: improve this
        for name in self.status:
            self.dec_status(name)
