from typing import Iterable

from game.entities.actors.monsters.base import Monster


class MonsterGroup(list):
    def __init__(self, iterable: Iterable[Monster]):
        if not all(isinstance(item, Monster) for item in iterable):
            raise ValueError("All elements must be Monsters")

        super().__init__(iterable)

    def append(self) -> None:
        raise NotImplementedError

    def extend(self) -> None:
        raise NotImplementedError

    def insert(self) -> None:
        raise NotImplementedError

    def __add__(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return "\n".join([f"{idx}) {monster}" for idx, monster in enumerate(self)])
