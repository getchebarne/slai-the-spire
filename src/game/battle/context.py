import random
from typing import Optional

from game.battle.comm import BattleView
from game.battle.pipelines.char import CharacterPipeline
from game.battle.pipelines.monster import MonsterPipeline
from game.battle.state import BattleState
from game.battle.systems.draw_card import DrawCard
from game.battle.systems.play_card import PlayCard
from game.battle.systems.resolve_target import ResolveTarget
from game.entities.actors.char import Character
from game.entities.actors.monster import MonsterCollection
from game.entities.cards.deck import Deck
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand


MAX_MONSTERS = 5


class BattleContext:
    def __init__(
        self,
        char: Character,
        monsters: MonsterCollection,
        deck: Deck,
        disc_pile: Optional[DiscardPile] = None,
        draw_pile: Optional[DrawPile] = None,
        hand: Optional[Hand] = None,
    ):
        if len(monsters) > MAX_MONSTERS:
            raise ValueError(f"Can't have more than {MAX_MONSTERS} monsters")

        self.char = char
        self.monsters = monsters
        self.deck = deck

        # Initialize as empty unless initial values are provided
        self.disc_pile = disc_pile if disc_pile else DiscardPile([])
        self.hand = hand if hand else Hand([])

        # Initialize as deck (shuffled)
        random.shuffle(deck.cards)
        self.draw_pile = (
            draw_pile if draw_pile else DrawPile(deck.cards)
        )  # TODO: shuffle

        # Set initial state
        self.state = BattleState.DEFAULT

        # Setup systems
        self._setup()

    def _setup(self) -> None:
        # Set initial state
        self.state = BattleState.DEFAULT

        # Pipelines
        self._char_pipe = CharacterPipeline()
        self._monster_pipe = MonsterPipeline()

        # Systems
        self._draw_card = DrawCard(self.disc_pile, self.draw_pile, self.hand)
        self._play_card = PlayCard(self.char, self.disc_pile, self.hand)
        self._resolve_target = ResolveTarget()

    def _setup_systems(self) -> None:
        # Draw card. TODO: remove?
        self._draw_card = DrawCard(self.disc_pile, self.draw_pile, self.hand)

    def is_over(self) -> bool:
        return self.char.health.current <= 0 or all(
            [monster.health.current <= 0 for monster in self.monsters]
        )

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

    def _process_monster_turn(self) -> None:
        for monster in self.monsters:
            effects = monster.move
            for effect in effects:
                # TODO: set monster's target properly
                self._process_effect(effect, self.char)

    def view(self) -> BattleView:
        return BattleView(
            self.state,
            self.char,
            self.monsters,
            self.disc_pile,
            self.draw_pile,
            self.hand,
        )
