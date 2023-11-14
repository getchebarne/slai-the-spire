from typing import Iterable, Iterator, List

from game.effects.relic import RelicEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.relics.base import BaseRelic


class RelicGroup(list):
    def __init__(self, iterable: Iterable[BaseRelic] = []):
        if not all(isinstance(item, BaseRelic) for item in iterable):
            raise ValueError(f"All elements must be instances of {BaseRelic}")

        super().__init__(iterable)

    def on_turn_end(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        effects = []
        for relic in self:
            effects.extend(relic.on_turn_end(char, monsters))

        return effects

    def on_turn_start(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        effects = []
        for relic in self:
            effects.extend(relic.on_turn_start(char, monsters))

        return effects

    def on_battle_end(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        effects = []
        for relic in self:
            effects.extend(relic.on_battle_end(char, monsters))

        return effects

    def on_battle_start(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        effects = []
        for relic in self:
            effects.extend(relic.on_battle_start(char, monsters))

        return effects

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
