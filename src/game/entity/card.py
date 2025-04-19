from dataclasses import dataclass

from src.game.combat.effect import Effect
from src.game.entity.base import EntityBase


@dataclass
class EntityCard(EntityBase):
    name: str
    cost: int
    effects: list[Effect]
