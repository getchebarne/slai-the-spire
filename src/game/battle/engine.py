from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from game.lib.card import card_lib
from game.lib.monster import monster_lib
from game.lib.move import move_lib
from game.lib.modifier import modifier_lib
from game.pipeline.pipeline import EffectPipeline
from game.context import Context
from game.context import BattleState


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
    def __init__(self, agent: BaseAgent, context: Context):
        self.agent = agent
        self.context = context

        # TODO: fix
        self.effect_pipeline = EffectPipeline()

        self._setup()

    def _setup(self) -> None:
        # Set state
        self.context.state = BattleState.DEFAULT

        # Fill draw pile
        self.context.draw_pile = self.context.deck.copy()
        random.shuffle(self.context.draw_pile)

        # Get first move from monsters
        for monster_id, monster_data in self.context.get_monsters():
            monster_ai = monster_lib[monster_data.name].ai
            self.context.monster_moves[monster_id] = monster_ai.first_move_name()

        # TODO: register relics?

    def _select_card(self, card_idx: int) -> None:
        # Get card from hand
        card_name = self.context.hand[card_idx]
        card_info = card_lib[card_name]

        # Check the player has enough energy to play the card
        if self.context.energy.current < card_info.card_cost:
            raise ValueError(f"Can't play {card_name} with {self.context.energy.current} energy")

        # Set active card
        self.context.active_card_idx = card_idx

    def _char_turn_start(self) -> None:
        # Draw cards from draw pile
        for _ in range(NUM_CARDS_DRAWN_PER_TURN):
            self._draw_one_card()

        # Reset energy
        self.context.energy.current = self.context.energy.max

        # Queue modifier effects
        char_id, _ = self.context.get_char()
        for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
            if entity_id == char_id:
                modifier_entry = modifier_lib[modifier_name]
                effects = modifier_entry.modifier_logic.at_start_of_turn(entity_id, stacks)
                self.effect_pipeline(self.context, effects)

        # Reset block
        self.context.get_char()[1].current_block = 0

    def _char_turn_end(self) -> None:
        # Queue modifier effects
        char_id, _ = self.context.get_char()
        for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
            if entity_id == char_id:
                modifier_entry = modifier_lib[modifier_name]
                effects = modifier_entry.modifier_logic.at_end_of_turn(entity_id, stacks)
                self.effect_pipeline(self.context, effects)

                # Decrease stacks if the modifier stacks duration. TODO: remove if 0
                if modifier_entry.modifier_stacks_duration:
                    self.context.entity_modifiers[(entity_id, modifier_name)] = max(0, stacks - 1)

        # Discard hand
        self.context.disc_pile.extend(self.context.hand)
        self.context.hand = []

    def _draw_one_card(self) -> None:
        # If the draw pile is empty, shuffle the discard pile and send it to the draw pile
        if len(self.context.draw_pile) == 0:
            random.shuffle(self.context.disc_pile)
            self.context.draw_pile = self.context.disc_pile
            self.context.disc_pile = []

        # Draw one card from draw pile
        self.context.hand.append(self.context.draw_pile[0])
        self.context.draw_pile = self.context.draw_pile[1:]

    def _monsters_turn_start(self) -> None:
        for monster_id, _ in self.context.get_monsters():
            # Queue modifier effects
            for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
                if entity_id == monster_id:
                    modifier_entry = modifier_lib[modifier_name]
                    effects = modifier_entry.modifier_logic.at_start_of_turn(entity_id, stacks)
                    self.effect_pipeline(self.context, effects)

            # Reset block
            self.context.entities[monster_id].current_block = 0

    def _monsters_turn_end(self) -> None:
        # Update monsters' moves
        for monster_id, monster_data in self.context.get_monsters():
            # Queue modifier effects
            for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
                if entity_id == monster_id:
                    modifier_entry = modifier_lib[modifier_name]
                    effects = modifier_entry.modifier_logic.at_end_of_turn(entity_id, stacks)
                    self.effect_pipeline(self.context, effects)

                    # Decrease stacks if the modifier stacks duration. TODO: remove if 0
                    if modifier_entry.modifier_stacks_duration:
                        self.context.entity_modifiers[(entity_id, modifier_name)] = max(
                            0, stacks - 1
                        )

            # Get monster's AI
            monster_ai = monster_lib[monster_data.name].ai

            # Get monster's current move
            current_move_name = self.context.monster_moves[monster_id]

            # Set monster's next move
            next_move_name = monster_ai.next_move_name(current_move_name)
            self.context.monster_moves[monster_id] = next_move_name

    def _play_card(self, monster_idx: Optional[int] = None) -> None:
        if self.context.active_card_idx is None:
            raise ValueError("No active card to play")

        # Get active card's effects
        card_name = self.context.hand[self.context.active_card_idx]
        card_info = card_lib[card_name]
        effects = card_info.card_logic.use(self.context, monster_idx)

        # Apply targeted effects
        self.effect_pipeline(self.context, effects)

        # Remove card from hand and send it to the draw pile.
        # TODO: cards can also be exhausted
        self.context.hand.remove(card_name)
        self.context.disc_pile.append(card_name)

        # Substract energy spent
        self.context.energy.current -= card_info.card_cost

        # Clear active card
        self.context.active_card_idx = None

    def _monsters_turn(self) -> None:
        # TODO: find way to improve this
        for monster_id, monster_data in self.context.get_monsters():
            # Get monster's move logic
            current_move_name = self.context.monster_moves[monster_id]
            move_logic = move_lib[(monster_data.name, current_move_name)]

            # Get effects
            effects = move_logic.use(self.context, monster_id)
            self.effect_pipeline(self.context, effects)

    def _char_turn(self) -> None:
        while not self.is_over():
            # Get action from agent
            action = self.agent.select_action(self.context)

            # End turn TODO: maybe add enemy turn state?
            if action.type == ActionType.END_TURN:
                self.context.state = BattleState.DEFAULT
                break

            # Default state
            if self.context.state == BattleState.DEFAULT:
                self._handle_default_state(action)
                continue

            # Await target state
            if self.context.state == BattleState.AWAIT_TARGET:
                self._handle_await_target_state(action)

    def _handle_default_state(self, action: Action) -> None:
        if action.type == ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: SELECT_TARGET in DEFAULT state")

        # TODO: add potion usage support
        if action.type != ActionType.SELECT_CARD:
            raise ValueError(f"Undefined action type {action.type}")

        self._select_card(action.index)
        self.context.state = BattleState.AWAIT_TARGET

    def _handle_await_target_state(self, action: Action) -> None:
        if action.type != ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: Expected SELECT_TARGET in AWAIT_TARGET state")

        # TODO: add potion usage support
        self._play_card(action.index)
        self.context.state = BattleState.DEFAULT

    def run(self) -> None:
        while not self.is_over():
            # Character's turn
            self._char_turn_start()
            self._char_turn()
            self._char_turn_end()

            # Monsters' turn
            self._monsters_turn_start()
            self._monsters_turn()
            self._monsters_turn_end()

    def is_over(self) -> bool:
        return self.context.get_char()[1].current_health <= 0 or all(
            monster_data.current_health <= 0 for _, monster_data in self.context.get_monsters()
        )
