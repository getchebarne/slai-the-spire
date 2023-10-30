from dataclasses import dataclass

from game.effects.base import BaseEffect


@dataclass
class MonsterEffect(BaseEffect):
    frail: int = 0
