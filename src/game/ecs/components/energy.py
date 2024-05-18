from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.base import BaseComponent


@dataclass
class EnergyComponent(BaseComponent):
    max: int
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max
