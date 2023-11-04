from abc import ABC
from dataclasses import dataclass

from game.entities.actors.base import BaseActor


# TODO: maybe initialize values with `None` instead of `0`
@dataclass
class BaseEffect(ABC):
    source: BaseActor
    target: BaseActor
    damage: int = 0
    block: int = 0
    weak: int = 0

    # TODO: add more consistency checks
    def __post_init__(self):
        if self.damage and self.block:
            raise ValueError("An effect can't have both damage and block")
