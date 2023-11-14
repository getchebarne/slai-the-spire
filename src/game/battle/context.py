from typing import Generator, List, Optional

from game.battle.systems.draw_card import DrawCard
from game.effects.base import BaseEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.deck import Deck
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand
from game.entities.relics.group import RelicGroup


MAX_MONSTERS = 5


class BattleContext:
    def __init__(
        self,
        char: Character,
        monsters: MonsterGroup,
        deck: Deck,
        disc_pile: DiscardPile = DiscardPile([]),
        draw_pile: Optional[DrawPile] = None,
        hand: Hand = Hand([]),
        relics: RelicGroup = RelicGroup([]),
    ):
        if len(monsters) > MAX_MONSTERS:
            raise ValueError(f"Can't have more than {MAX_MONSTERS} monsters")

        self.char = char
        self.monsters = monsters
        self.deck = deck
        self.disc_pile = disc_pile
        self.hand = hand
        self.relics = relics

        # Initialize as deck (shuffled)
        self.draw_pile = draw_pile if draw_pile is not None else DrawPile(deck.cards).shuffle()

        # Setup systems
        self._setup()

    def _setup(self) -> None:
        # Systems. TODO: re-evaluate this
        self._draw_card = DrawCard(self.disc_pile, self.draw_pile, self.hand)

    def is_over(self) -> bool:
        return self.char.health.current <= 0 or all(
            [monster.health.current <= 0 for monster in self.monsters]
        )

    def battle_start(self) -> Generator[List[BaseEffect], None, None]:
        # Relic effects
        for relic in self.relics:
            yield relic.on_battle_start(self.char, self.monsters)

    def char_turn_start(self) -> List[BaseEffect]:
        # Reset block & energy
        self.char.block.current = 0
        self.char.energy.current = self.char.energy.max

        # Draw 5 cards
        self._draw_card(5)

        # Return modifier effects
        return self.char.on_turn_start(self.char, self.monsters)

    def monsters_turn_start(self) -> None:
        # Reset block
        for monster in self.monsters:
            monster.block.current = 0

    def char_turn_end(self) -> None:
        # Discard cards in hand
        self.disc_pile.cards.extend(self.hand.cards)
        self.hand.cards = []

    def monsters_turn_end(self) -> None:
        for monster in self.monsters:
            monster.update_move()
