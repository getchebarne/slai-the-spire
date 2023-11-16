from __future__ import annotations

import random
from typing import List

from game.entities.cards.base import BaseCard


class DrawPile:
    def __init__(self, cards: List[BaseCard] = []):
        self.cards = cards

    def shuffle(self) -> DrawPile:
        # TODO: seed
        random.shuffle(self.cards)
        return self
