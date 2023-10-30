from typing import Tuple

from game.battle.systems.base import BaseSystem
from game.effects.base import TargetType
from game.entities.actors.char import Character
from game.entities.cards.base import BaseCard
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.hand import Hand


class PlayCard(BaseSystem):
    def __init__(self, char: Character, disc_pile: DiscardPile, hand: Hand):
        self.char = char
        self.disc_pile = disc_pile
        self.hand = hand

    # TODO: trigger on-card-play events (e.g., sharp hide, kunai)
    def __call__(self, card_idx: int) -> Tuple[BaseCard, bool]:
        # Get card from hand
        card = self.hand[card_idx]

        # Check the player has enough energy to play the card
        if self.char.energy.current < card.cost:
            raise ValueError(
                f"Can't play {card} with {self.char.energy.current} energy"
            )

        # Substract energy spent. TODO: this shouldn't be here
        self.char.energy.current -= card.cost

        # Remove card from hand and send it to the discard pile
        # TODO: add methods to remove and append cards?
        self.hand.cards.remove(card)
        self.disc_pile.cards.append(card)

        # Return card to be played & wether it requires additional target input
        return card, self._requires_target(card)

    # TODO: try to find better place for this logic
    @staticmethod
    def _requires_target(card: BaseCard) -> bool:
        return any([effect.target_type == TargetType.SINGLE for effect in card.effects])
