from typing import List

from game.entities.cards.base import BaseCard


class Hand:
    def __init__(self, cards: List[BaseCard]):
        self.cards = cards

    def __getitem__(self, idx: int) -> BaseCard:
        return self.cards[idx]

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return " | ".join([str(card) for card in self.cards])
