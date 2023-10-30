from dataclasses import dataclass

from game.effects.base import BaseEffect


@dataclass
class CardEffect(BaseEffect):
    draw: int = 0
