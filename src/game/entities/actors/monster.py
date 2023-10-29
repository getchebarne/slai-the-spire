from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List

from game.effects.monster import MonsterEffect
from game.entities.actors.base import BaseActor
from game.entities.actors.base import Block
from game.entities.actors.base import Buffs
from game.entities.actors.base import Debuffs
from game.entities.actors.base import Health


# TODO: probably set defaults to `None`
# TODO: apply intent correction based on buffs / debuffs (e.g., weak)?
@dataclass
class Intent:
    damage: int = 0
    instances: int = 0
    block: bool = False
    buff: bool = False
    debuff: bool = False
    strong_debuff: bool = False
    escape: bool = False
    asleep: bool = False
    stunned: bool = False
    unknown: bool = False

    def __str__(self) -> str:
        # TODO: add support for other intents
        str_ = ""
        if self.damage:
            str_ = f"{str_}\U0001F5E1 {self.damage}"

        if self.instances > 1:
            str_ = f"{str_}x{self.instances}"

        return str_


class Monster(BaseActor):
    moves: Dict[str, List[MonsterEffect]] = dict()

    def __init__(
        self,
        health: Health,
        block: Block,
        buffs: Buffs,
        debuffs: Debuffs,
    ) -> None:
        super().__init__(health, block, buffs, debuffs)

        # Check if `moves` is defined
        if not hasattr(self.__class__, "moves") or not self.__class__.moves:
            raise NotImplementedError(
                "Subclasses of `Monster` must define a `moves` class variable"
            )

        # Set initial move
        self._set_first_move()

    @abstractmethod
    def update_move(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_first_move(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _move_to_intent(move: List[MonsterEffect]) -> Intent:
        # TODO: improve this
        damage = None
        instances = 0
        block = False
        debuff = False
        for effect in move:
            if effect.damage:
                if damage is None:
                    damage = effect.damage
                else:
                    if damage != effect.damage:
                        # TODO: should this check be here?
                        raise ValueError(
                            "A monster can't have instances with different damage"
                        )

                instances += 1

            if effect.block:
                block = True

            if effect.frail:
                debuff = True

            if effect.weak:
                debuff = True

        return Intent(damage=damage, instances=instances, block=block, debuff=debuff)

    def __str__(self) -> str:
        # TODO: improve intent
        # Get BaseActor string
        base_str = super().__str__()

        # Append intent
        return f"{base_str} {self.intent}"


class MonsterCollection(list):
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
