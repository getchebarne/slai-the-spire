import random
from typing import Optional

from game.battle.pipeline.pipeline import EffectPipeline
from game.battle.systems.draw_card import DrawCard
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.deck import Deck
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand
from game.entities.relics.base import Relics


MAX_MONSTERS = 5


class BattleContext:
    def __init__(
        self,
        char: Character,
        monsters: MonsterGroup,
        deck: Deck,
        disc_pile: Optional[DiscardPile] = None,
        draw_pile: Optional[DrawPile] = None,
        hand: Optional[Hand] = None,
        relics: Optional[Relics] = None,
    ):
        if len(monsters) > MAX_MONSTERS:
            raise ValueError(f"Can't have more than {MAX_MONSTERS} monsters")

        self.char = char
        self.monsters = monsters
        self.deck = deck

        # Initialize as empty unless initial values are provided
        self.disc_pile = disc_pile if disc_pile else DiscardPile([])
        self.hand = hand if hand else Hand([])
        self.relics = relics if relics is not None else Relics()

        # Initialize as deck (shuffled)
        random.shuffle(deck.cards)
        self.draw_pile = (
            draw_pile if draw_pile else DrawPile(deck.cards)
        )  # TODO: shuffle

        # Setup systems
        self._setup()

    def _setup(self) -> None:
        # Pipeline. TODO: maybe move to `BattleEngine`
        self.pipeline = EffectPipeline()

        # Systems
        self._draw_card = DrawCard(self.disc_pile, self.draw_pile, self.hand)

    def is_over(self) -> bool:
        return self.char.health.current <= 0 or all(
            [monster.health.current <= 0 for monster in self.monsters]
        )

    def battle_start(self) -> None:
        for relic in self.relics:
            relic_effects = relic.on_battle_start(self.char, self.monsters)
            if len(relic_effects) > 0:
                self.pipeline(relic_effects)

    def char_turn_start(self) -> None:
        # Reset block & energy
        self.char.block.current = 0
        self.char.energy.current = self.char.energy.max

        # Draw 5 cards
        self._draw_card(5)

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
