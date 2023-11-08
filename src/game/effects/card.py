from dataclasses import dataclass
from typing import Optional

from game.effects.base import BaseEffect


@dataclass
class CardEffect(BaseEffect):
    draw: Optional[int] = None
