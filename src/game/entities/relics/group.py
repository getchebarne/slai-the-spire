from typing import Iterable
from typing import Iterator

from game.entities.relics.base import BaseRelic


class RelicGroup(list):
    def __init__(self, iterable: Iterable[BaseRelic] = []):
        if not all(isinstance(item, BaseRelic) for item in iterable):
            raise ValueError(f"All elements must be instances of {BaseRelic}")

        super().__init__(iterable)

    def append(self) -> None:
        raise NotImplementedError

    def extend(self) -> None:
        raise NotImplementedError

    def insert(self) -> None:
        raise NotImplementedError

    def __add__(self) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> BaseRelic:
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[BaseRelic]:
        return super().__iter__()
