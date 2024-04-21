from dataclasses import dataclass
from typing import List

from src.game.core.card import Card


@dataclass
class DrawPile:
    cards: List[Card]
