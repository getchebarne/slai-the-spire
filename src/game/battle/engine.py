from typing import List, Optional

from agents.base import BaseAgent
from game.battle.comm import ActionType
from game.battle.comm import BattleView
from game.battle.context import BattleContext
from game.battle.drawer import BattleDrawer
from game.battle.pipelines.base import TargetedEffect
from game.battle.state import BattleState
from game.effects.base import BaseEffect
from game.effects.base import TargetType
from game.entities.actors.base import BaseActor
from game.entities.cards.base import BaseCard


class BattleEngine:
    def __init__(
        self,
        agent: BaseAgent,
        context: BattleContext,
        drawer: Optional[BattleDrawer] = None,
    ):
        self.agent = agent
        self.context = context
        self.drawer = drawer

        # Setup
        self._state = BattleState.DEFAULT
        self._active_card: BaseCard = None

    def _select_card(self, card_idx: int) -> None:
        # Get card from hand
        card = self.context.hand[card_idx]

        # Check the player has enough energy to play the card
        if self.context.char.energy.current < card.cost:
            raise ValueError(
                f"Can't play {card} with {self.char.energy.current} energy"
            )

        # Set active card
        self._active_card = card

    def _play_card(self, monster_idx: Optional[int] = None) -> None:
        # Remove card from hand and send it to the draw pile
        self.context.hand.cards.remove(self._active_card)
        self.context.disc_pile.cards.append(self._active_card)

        # Substract energy
        self.context.char.energy.current -= self._active_card.cost

        # Get target monster if there's any
        monster = (
            self.context.monsters[monster_idx] if monster_idx is not None else None
        )
        # Get targeted effects
        targeted_effects = [
            targeted_effect
            for effect in self._active_card.effects
            for targeted_effect in self._resolve_target(
                effect, self.context.char, monster
            )
        ]
        # Apply targeted effects.
        for targeted_effect in targeted_effects:
            self.context.char_pipe(targeted_effect)

    def _resolve_target(
        self, effect: BaseEffect, source: BaseActor, target: Optional[BaseActor] = None
    ) -> List[TargetedEffect]:
        if effect.target_type == TargetType.SELF:
            return [TargetedEffect(effect, source, source)]

        if effect.target_type == TargetType.ALL_MONSTERS:
            return [
                TargetedEffect(effect, source, monster)
                for monster in self.context.monsters
            ]

        if effect.target_type == TargetType.SINGLE:
            if target is None:
                raise ValueError(
                    "Argument `target` can't be None for effects with target type "
                    f"{TargetType.SINGLE}"
                )
            return [TargetedEffect(effect, source, target)]

    def _monsters_turn(self) -> None:
        # TODO: find way to improve this
        for monster in self.context.monsters:
            for effect in monster.move:
                targeted_effects = self._resolve_target(
                    effect, monster, self.context.char
                )
                for targeted_effect in targeted_effects:
                    self.context.monster_pipe(targeted_effect)
                    if self.context.is_over():
                        return

    def _char_turn(self) -> None:
        while not self.context.is_over():
            # TODO: improve drawing logic
            if self.drawer:
                self.drawer(self.view())

            # Get action from agent
            action_type, action_idx = self.agent.select_action(self.view())

            # End turn TODO: maybe add enemy turn state?
            if action_type == ActionType.END_TURN:
                self._state = BattleState.DEFAULT
                break

            # Default state
            if self._state == BattleState.DEFAULT:
                self._handle_default_state(action_type, action_idx)
                continue

            # Await target state
            if self._state == BattleState.AWAIT_TARGET:
                self._handle_await_target_state(action_type, action_idx)

    def _handle_default_state(self, action_type: ActionType, action_idx: int) -> None:
        if action_type == ActionType.SELECT_TARGET:
            raise ValueError("Invalid action type: SELECT_TARGET in DEFAULT state")

        # TODO: add potion usage support
        if action_type != ActionType.SELECT_CARD:
            raise ValueError(f"Undefined action type {action_type}")

        self._select_card(action_idx)
        self._state = BattleState.AWAIT_TARGET

    def _handle_await_target_state(
        self, action_type: ActionType, action_idx: int
    ) -> None:
        if action_type != ActionType.SELECT_TARGET:
            raise ValueError(
                "Invalid action type: Expected SELECT_TARGET in AWAIT_TARGET state"
            )

        # TODO: add potion usage support
        self._play_card(action_idx)
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
        while not self.context.is_over():
            # Character's turn
            self.context.char_turn_start()
            self._char_turn()
            self.context.char_turn_end()

            # Monsters' turn
            self.context.monsters_turn_start()
            self._monsters_turn()
            self.context.monsters_turn_end()
