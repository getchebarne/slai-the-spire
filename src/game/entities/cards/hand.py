from typing import List, Optional

from game.entities.cards.base import BaseCard


class Hand:
    def __init__(self, cards: List[BaseCard] = [], active_card: Optional[BaseCard] = None):
        if active_card is not None and active_card not in cards:
            raise ValueError(f"Card {active_card} is not in the specified input `cards`")

        self.cards = cards
        self.active_card = active_card

    def set_active_card(self, idx: int) -> None:
        self.active_card = self.cards[idx]

    def clear_active_card(self) -> None:
        self.active_card = None

    def __getitem__(self, idx: int) -> BaseCard:
        return self.cards[idx]

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return " / ".join(
            [
                f"\033[92m{str(card)}\033[0m" if card == self.active_card else str(card)
                for card in self.cards
            ]
        )
