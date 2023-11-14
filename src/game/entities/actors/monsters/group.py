from typing import Iterable, Iterator

from game.entities.actors.monsters.base import Monster


class MonsterGroup(list):
    def __init__(self, iterable: Iterable[Monster]):
        if not all(isinstance(item, Monster) for item in iterable):
            raise ValueError(f"All elements must be instances of {Monster}")

        super().__init__(iterable)

    def append(self) -> None:
        raise NotImplementedError

    def extend(self) -> None:
        raise NotImplementedError

    def insert(self) -> None:
        raise NotImplementedError

    def __add__(self) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Monster:
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[Monster]:
        return super().__iter__()

    def __str__(self) -> str:
        return "\n".join([str(monster) for monster in self])
