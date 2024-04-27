from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from src.game.context import BattleState
from src.game.context import Context
from src.game.lib.card import card_lib
from src.game.lib.modifier import modifier_lib
from src.game.lib.monster import monster_lib
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.lib.relic import relic_lib
from src.game.lib.move import move_lib
from src.game.pipeline.pipeline import EffectPipeline


if TYPE_CHECKING:
    from src.agents.base import BaseAgent


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

        # Get first move from monsters. TODO: improve
        for monster_id, monster_data in self.context.get_monsters():
            if self.context.monster_moves[monster_id] is None:
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
        # Draw cards from draw pile. TODO: make source entity id and target entity id optional in
        # Effect
        self.effect_pipeline(
            self.context, [Effect(None, None, EffectType.DRAW_CARD, NUM_CARDS_DRAWN_PER_TURN)]
        )
        # Reset energy
        self.context.energy.current = self.context.energy.max

        # Queue modifier effects
        for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
            if entity_id == self.context.CHAR_ENTITY_ID:
                modifier_entry = modifier_lib[modifier_name]
                effects = modifier_entry.modifier_logic.at_start_of_turn(entity_id, stacks)
                self.effect_pipeline(self.context, effects)

        # Reset block
        self.context.entities[self.context.CHAR_ENTITY_ID].current_block = 0

    def _char_turn_end(self) -> None:
        # Queue modifier effects
        for (entity_id, modifier_name), stacks in self.context.entity_modifiers.items():
            if entity_id == self.context.CHAR_ENTITY_ID:
                modifier_entry = modifier_lib[modifier_name]
                effects = modifier_entry.modifier_logic.at_end_of_turn(entity_id, stacks)
                self.effect_pipeline(self.context, effects)

                # Decrease stacks if the modifier stacks duration. TODO: remove if 0
                if modifier_entry.modifier_stacks_duration:
                    self.context.entity_modifiers[(entity_id, modifier_name)] = max(0, stacks - 1)

        # Discard hand
        self.context.disc_pile.extend(self.context.hand)
        self.context.hand = []

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

    def _play_card(self, monster_entity_id: Optional[int] = None) -> None:
        if self.context.active_card_idx is None:
            raise ValueError("No active card to play")

        # Get active card's effects
        card_name = self.context.hand[self.context.active_card_idx]
        card_info = card_lib[card_name]
        effects = card_info.card_logic.use(self.context, monster_entity_id)

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
            self._execute_monster_move(monster_id)

    def _execute_monster_move(self, monster_entity_id: int) -> None:
        # Get monster's move logic
        monster_name = self.context.entities[monster_entity_id].name
        current_move_name = self.context.monster_moves[monster_entity_id]
        move_logic = move_lib[(monster_name, current_move_name)]

        # Get effects
        effects = move_logic.use(self.context, monster_entity_id)
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
        # Battle start
        self.battle_start()

        while not self.is_over():
            self.run_one_turn()

        # Battle end
        self.battle_end()

    def battle_start(self) -> None:
        # Queue relic effects
        for relic in self.context.relics:
            relic_entry = relic_lib[relic]
            effects = relic_entry.relic_logic.at_start_of_battle(self.context)
            self.effect_pipeline(self.context, effects)

    def battle_end(self) -> None:
        # Queue relic effects
        for relic in self.context.relics:
            relic_entry = relic_lib[relic]
            effects = relic_entry.relic_logic.at_end_of_battle(self.context)
            self.effect_pipeline(self.context, effects)

    # TODO: revise name
    def run_one_turn(self) -> None:
        # Character's turn
        self.run_char_turn()

        # Monsters' turn
        self.run_monsters_turn()

    def run_char_turn(self) -> None:
        self._char_turn_start()
        self._char_turn()
        self._char_turn_end()

    def run_monsters_turn(self) -> None:
        self._monsters_turn_start()
        self._monsters_turn()
        self._monsters_turn_end()

    def is_over(self) -> bool:
        return self.context.entities[self.context.CHAR_ENTITY_ID].current_health <= 0 or all(
            monster_data.current_health <= 0 for _, monster_data in self.context.get_monsters()
        )
