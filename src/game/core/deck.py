from dataclasses import dataclass
from typing import List

from game.core.card import Card


@dataclass
class Deck:
    cards: List[Card]
