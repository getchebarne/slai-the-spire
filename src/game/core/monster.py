from dataclasses import dataclass
from typing import Optional

from src.game.core.entity import Entity


@dataclass
class Monster(Entity):
    current_move_name: Optional[str] = None
