from game.battle.systems.base import BaseSystem
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand


class DrawCard(BaseSystem):
    def __init__(self, disc_pile: DiscardPile, draw_pile: DrawPile, hand: Hand):
        self.disc_pile = disc_pile
        self.draw_pile = draw_pile
        self.hand = hand

    def __call__(self, num: int) -> None:
        for _ in range(num):
            self._draw_one_card()

    def _draw_one_card(self) -> None:
        if not self.draw_pile.cards:
            # Shuffle discard pile into draw pile. TODO: add shuffle method?
            self.draw_pile.cards = self.disc_pile.cards
            self.draw_pile.shuffle()
            self.disc_pile.cards = []

        # Draw first card from draw pile & append it to the player's hand
        card = self.draw_pile.cards[0]
        self.hand.cards.append(card)

        # Remove card from draw pile
        self.draw_pile.cards.remove(card)
