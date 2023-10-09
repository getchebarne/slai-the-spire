from dataclasses import dataclass
from typing import List


from entities.base import BaseEntity
from entities.card import Card


@dataclass
class Deck(BaseEntity):
    cards: List[Card]

    def __len__(self) -> int:
        return len(self.cards)
