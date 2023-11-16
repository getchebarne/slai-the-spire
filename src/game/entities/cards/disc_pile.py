from typing import List

from game.entities.cards.base import BaseCard


class DiscardPile:
    def __init__(self, cards: List[BaseCard] = []):
        self.cards = cards
