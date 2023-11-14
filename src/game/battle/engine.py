from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from game.battle.context import BattleContext
from game.battle.drawer import BattleDrawer
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.base import BaseCard
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand
from game.pipeline.pipeline import EffectPipeline


if TYPE_CHECKING:
    from agents.base import BaseAgent


class ActionType(Enum):
    SELECT_CARD = 0
    SELECT_TARGET = 1
    END_TURN = 2


@dataclass
class Action:
    type: ActionType
    index: Optional[int]


class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1


@dataclass
class BattleView:
    state: BattleState
    active_card: BaseCard
    char: Character
    monsters: MonsterGroup
    disc_pile: DiscardPile
    draw_pile: DrawPile
    hand: Hand


class BattleEngine:
    def __init__(
        self,
        agent: BaseAgent,
        context: BattleContext,
        effect_pipeline: EffectPipeline = EffectPipeline(),
        drawer: Optional[BattleDrawer] = BattleDrawer(),
    ):
        self.agent = agent
        self.context = context
        self.effect_pipeline = effect_pipeline
        self.drawer = drawer

        self._setup()

    def _setup(self) -> None:
        # Set state & active card
        self._state: BattleState = BattleState.DEFAULT
        self._active_card: BaseCard = None

        # Register relics
        for relic in self.context.relics:
            if relic.step is not None:
                self.effect_pipeline.add_step(relic.step)

    def _select_card(self, card_idx: int) -> None:
        # Get card from hand
        card = self.context.hand[card_idx]

        # Check the player has enough energy to play the card
        if self.context.char.energy.current < card.cost:
            raise ValueError(f"Can't play {card} with {self.char.energy.current} energy")

        # Set active card
        self._active_card = card

    def _play_card(self, monster_idx: Optional[int] = None) -> None:
        # Get targeted effects
        effects = self._active_card.use(self.context.char, self.context.monsters, monster_idx)

        # Apply targeted effects
        self.effect_pipeline(effects)

        # Remove card from hand and send it to the draw pile.
        # TODO: cards can also be exhausted
        self.context.hand.cards.remove(self._active_card)
        self.context.disc_pile.cards.append(self._active_card)

        # Substract energy
        self.context.char.energy.current -= self._active_card.cost

    def _monsters_turn(self) -> None:
        # TODO: find way to improve this
        for monster in self.context.monsters:
            effects = monster.execute_move(self.context.char, self.context.monsters)
            self.effect_pipeline(effects)

    def _char_turn(self) -> None:
        while not self.context.is_over():
            # TODO: improve drawing logic
            if self.drawer is not None:
                self.drawer(self.view())

            # Get action from agent
            action = self.agent.select_action(self.view())

            # End turn TODO: maybe add enemy turn state?
            if action.type == ActionType.END_TURN:
                self._state = BattleState.DEFAULT
                break

            # Default state
            if self._state == BattleState.DEFAULT:
                self._handle_default_state(action)
                continue

            # Await target state
            if self._state == BattleState.AWAIT_TARGET:
                self._handle_await_target_state(action)

    def _handle_default_state(self, action: Action) -> None:
        if action.type == ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: SELECT_TARGET in DEFAULT state")

        # TODO: add potion usage support
        if action.type != ActionType.SELECT_CARD:
            raise ValueError(f"Undefined action type {action.type}")

        self._select_card(action.index)
        self._state = BattleState.AWAIT_TARGET

    def _handle_await_target_state(self, action: Action) -> None:
        if action.type != ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: Expected SELECT_TARGET in AWAIT_TARGET state")

        # TODO: add potion usage support
        self._play_card(action.index)
        self._state = BattleState.DEFAULT

    def view(self) -> BattleView:
        return BattleView(
            self._state,
            self._active_card,
            self.context.char,
            self.context.monsters,
            self.context.disc_pile,
            self.context.draw_pile,
            self.context.hand,
        )

    def run(self) -> None:
        if self.drawer is not None:
            self.drawer(self.view())

        # Battle start
        for effects in self.context.battle_start():
            self.effect_pipeline(effects)

        while not self.context.is_over():
            # Character's turn
            effects = self.context.char_turn_start()
            self.effect_pipeline(effects)

            self._char_turn()

            effects = self.context.char_turn_end()
            self.effect_pipeline(effects)

            # Monsters' turn
            effects = self.context.monsters_turn_start()
            self.effect_pipeline(effects)

            self._monsters_turn()

            effects = self.context.monsters_turn_end()
            self.effect_pipeline(effects)

        # Battle end
        for effects in self.context.battle_end():
            self.effect_pipeline(effects)
