from dataclasses import dataclass

from src.game.entity.base import EntityBase


@dataclass
class EntityEnergy(EntityBase):
    max: int = 3
    current: int | None = None

    def __post_init__(self):
        if self.current is None:
            self.current = self.max
