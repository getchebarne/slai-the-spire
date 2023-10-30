from agents.base import BaseAgent
from game.battle.comm import ActionType
from game.battle.context import BattleContext
from game.battle.drawer import BattleDrawer
from game.battle.state import BattleState


class BattleEngine:
    def __init__(
        self,
        agent: BaseAgent,
        context: BattleContext,
        drawer: BattleDrawer,
        draw: bool = True,
    ):
        self.agent = agent
        self.context = context
        self.drawer = drawer
        self.draw = draw

    def _char_turn(self) -> None:
        # TODO: improve this
        while not self.context.is_over():
            if self.draw:
                self.drawer(self.context.view())

            # Get action from agent
            action_type, action_idx = self.agent.select_action(self.context.view())

            if action_type == ActionType.END_TURN:
                break

            if self.context.state == BattleState.DEFAULT:
                if action_type == ActionType.SELECT_TARGET:
                    raise ValueError(
                        "Invalid action type: SELECT_TARGET in DEFAULT state"
                    )

                if action_type == ActionType.PLAY_CARD:
                    card, requires_target = self.context._play_card(action_idx)

                if requires_target:
                    new_state = BattleState.AWAIT_TARGET
                else:
                    targeted_effects = self.context._resolve_target(
                        card.effects, self.context.char
                    )
                    for targeted_effect in targeted_effects:
                        self.context._char_pipe(
                            targeted_effect.effect, targeted_effect.target
                        )
                    new_state = BattleState.DEFAULT

            elif self.context.state == BattleState.AWAIT_TARGET:
                if action_type != ActionType.SELECT_TARGET:
                    raise ValueError(
                        "Invalid action type: Expected SELECT_TARGET in AWAIT_TARGET state"
                    )

                targeted_effects = self.context._resolve_target(
                    card.effects, self.context.char, self.context.monsters[action_idx]
                )
                for targeted_effect in targeted_effects:
                    self.context._char_pipe(
                        targeted_effect.effect, targeted_effect.target
                    )

                new_state = BattleState.DEFAULT

            self.context.state = new_state

    def _monsters_turn(self) -> None:
        for monster in self.context.monsters:
            effects = monster.move
            # TODO: set monster's target properly
            targeted_effects = self.context._resolve_target(
                effects, monster, self.context.char
            )
            for targeted_effect in targeted_effects:
                self.context._monster_pipe(
                    targeted_effect.effect, targeted_effect.target
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
