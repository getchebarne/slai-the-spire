from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseEffect(ABC):
    damage: Optional[int] = None
    block: Optional[int] = None
    weak: Optional[int] = None
    plus_str: Optional[int] = None

    # TODO: add more consistency checks
    def __post_init__(self):
        if self.damage and self.block:
            raise ValueError("An effect can't have both damage and block")
