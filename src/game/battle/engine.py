from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from game import ai
from game import context
from game.lib.card import card_lib
from game.lib.monster import monster_lib
from game.pipeline.pipeline import EffectPipeline


if TYPE_CHECKING:
    from agents.base import BaseAgent


NUM_CARDS_DRAWN_PER_TURN = 5


class ActionType(Enum):
    SELECT_CARD = 0
    SELECT_TARGET = 1
    END_TURN = 2


@dataclass
class Action:
    type: ActionType
    index: Optional[int]


class BattleEngine:
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        # TODO: fix
        self.effect_pipeline = EffectPipeline()

        self._setup()

    def _setup(self) -> None:
        # Set state
        context.state = context.BattleState.DEFAULT

        # Fill draw pile
        context.draw_pile = context.deck.copy()
        random.shuffle(context.draw_pile)

        # TODO: register relics?

    def _select_card(self, card_idx: int) -> None:
        # Get card from hand
        card_name = context.hand[card_idx]
        card_info = card_lib[card_name]

        # Check the player has enough energy to play the card
        if context.energy.current < card_info.card_cost:
            raise ValueError(f"Can't play {card_name} with {context.energy.current} energy")

        # Set active card
        context.active_card = card_name

    def _char_turn_start(self) -> None:
        # Draw cards from draw pile
        for _ in range(NUM_CARDS_DRAWN_PER_TURN):
            self._draw_one_card()

        # Reset energy
        context.energy.current = context.energy.max

        # Update monsters' moves. TODO: fix, use ai
        for monster in context.monsters:
            monster.current_move_name = random.choice(list(monster_lib[monster.name].moves.keys()))

    def _char_turn_end(self) -> None:
        # Discard hand
        context.disc_pile.extend(context.hand)
        context.hand = []

    def _draw_one_card(self) -> None:
        # If the draw pile is empty, shuffle the discard pile and send it to the draw pile
        if len(context.draw_pile) == 0:
            random.shuffle(context.disc_pile)
            context.draw_pile = context.disc_pile
            context.disc_pile = []

        # Draw one card from draw pile
        context.hand.append(context.draw_pile[0])
        context.draw_pile = context.draw_pile[1:]

    def _monsters_turn_start(self) -> None:
        pass

    def _monsters_turn_end(self) -> None:
        # Update monsters' moves
        for monster in context.monsters:
            # TODO: implement base ai class
            # TODO: write function to get ai class from monster name instead of using getattr
            monster_ai = getattr(ai, monster.name)
            monster.current_move_name = monster_ai.next_move_name(monster.current_move_name)

    def _play_card(self, monster_idx: Optional[int] = None) -> None:
        if context.active_card is None:
            raise ValueError("No active card to play")

        # Get active card's effects
        card_name = context.active_card
        card_info = card_lib[card_name]
        effects = card_info.card_logic.use(monster_idx)

        # Apply targeted effects
        self.effect_pipeline(effects)

        # Remove card from hand and send it to the draw pile.
        # TODO: cards can also be exhausted
        context.hand.remove(card_name)
        context.disc_pile.append(card_name)

        # Substract energy spent
        context.energy.current -= card_info.card_cost

        # Clear active card
        context.active_card = None

    def _monsters_turn(self) -> None:
        # TODO: find way to improve this
        for monster in context.monsters:
            move = monster_lib[monster.name].moves[monster.current_move_name]
            effects = move.use(monster)
            self.effect_pipeline(effects)

    def _char_turn(self) -> None:
        while not self.is_over():
            # Get action from agent
            action = self.agent.select_action()

            # End turn TODO: maybe add enemy turn state?
            if action.type == ActionType.END_TURN:
                context.state = context.BattleState.DEFAULT
                break

            # Default state
            if context.state == context.BattleState.DEFAULT:
                self._handle_default_state(action)
                continue

            # Await target state
            if context.state == context.BattleState.AWAIT_TARGET:
                self._handle_await_target_state(action)

    def _handle_default_state(self, action: Action) -> None:
        if action.type == ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: SELECT_TARGET in DEFAULT state")

        # TODO: add potion usage support
        if action.type != ActionType.SELECT_CARD:
            raise ValueError(f"Undefined action type {action.type}")

        self._select_card(action.index)
        context.state = context.BattleState.AWAIT_TARGET

    def _handle_await_target_state(self, action: Action) -> None:
        if action.type != ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: Expected SELECT_TARGET in AWAIT_TARGET state")

        # TODO: add potion usage support
        self._play_card(action.index)
        context.state = context.BattleState.DEFAULT

    def run(self) -> None:
        while not self.is_over():
            # Character's turn
            self._char_turn_start()
            self._char_turn()
            self._char_turn_end()

            # Monsters' turn
            self._monsters_turn()

    @staticmethod
    def is_over() -> bool:
        return context.char.health.current <= 0 or all(
            monster.health.current <= 0 for monster in context.monsters
        )
