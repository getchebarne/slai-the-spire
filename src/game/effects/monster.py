from dataclasses import dataclass
from typing import Optional

from game.effects.base import BaseEffect


@dataclass
class MonsterEffect(BaseEffect):
    frail: Optional[int] = None
