from typing import Callable

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.drawer import draw_combat
from src.game.combat.utils import does_card_require_target
from src.game.combat.utils import is_game_over
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat
from src.game.core.combat_state import CombatState
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.engine.effect_queue import add_to_bot
from src.game.engine.effect_queue import add_to_top
from src.game.engine.effect_queue import process_effect_queue
from src.game.engine.state import GameState


class InvalidActionError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _handle_select_entity(
    game_state: GameState, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.combat_state == CombatState.DEFAULT:
        if id_target not in game_state.entity_manager.id_cards_in_hand:
            raise InvalidActionError("Can only select cards in hand while in default state")

        # Get the selected card
        card = game_state.entity_manager.entities[id_target]

        # Check if there's enough energy to play it
        energy_current = game_state.entity_manager.entities[
            game_state.entity_manager.id_energy
        ].current
        if card.cost > energy_current:
            raise InvalidActionError(f"Can't select card {card} with {energy_current} energy")

        # If the card requires targeting, set it as active and return
        if does_card_require_target(card):
            return (
                [Effect(EffectType.CARD_ACTIVE_SET, id_target=id_target)],
                [],
            )

        # Else, play it right away. This is different from the original game's implementation,
        # where cards that don't need a target are still set as active and await the player's
        # confirmation
        return (
            [Effect(EffectType.CARD_PLAY, id_target=id_target)],
            [],
        )

    if game_state.combat_state == CombatState.AWAIT_TARGET_CARD:
        # Queue is empty in this state. First the card's target is set, then the active card is
        # cleared, then the card is played (it's effects are added to the top of the queue), and
        # finally the card's target is cleared
        return (
            [
                Effect(EffectType.TARGET_CARD_SET, id_target=id_target),
                Effect(EffectType.CARD_ACTIVE_CLEAR),
                Effect(EffectType.CARD_PLAY, id_target=game_state.entity_manager.id_card_active),
                Effect(EffectType.TARGET_CARD_CLEAR),
            ],
            [],
        )

    if game_state.combat_state == CombatState.AWAIT_TARGET_EFFECT:
        # An effect is added at the top of the queue to set the effect's target. The effect that
        # comes inmediately after is going to use this variable to resolve its target. For now,
        # an effect to clear the effect's target is added to the bottom of the queue, but TODO:
        # I think I can't escape adding the clear when the `id_effect_target` is consumed
        return (
            [Effect(EffectType.TARGET_EFFECT_CLEAR, id_target=id_target)],
            [Effect(EffectType.TARGET_EFFECT_SET, id_target=id_target)],
        )


def handle_action(game_state: GameState, action: Action) -> tuple[list[Effect], list[Effect]]:
    if action.type == ActionType.END_TURN:
        return (
            [Effect(EffectType.TURN_END, id_target=game_state.entity_manager.id_character)],
            [],
        )

    elif action.type == ActionType.SELECT_ENTITY:
        return _handle_select_entity(game_state, action.target_id)


def step(game_state: GameState, action: Action) -> None:
    # Handle action
    effects_bot, effects_top = handle_action(game_state, action)

    # Add new effects to the queue
    add_to_bot(game_state.effect_queue, *effects_bot)
    add_to_top(game_state.effect_queue, *effects_top)

    # Process round
    process_effect_queue(game_state.entity_manager, game_state.effect_queue)

    # Set new state
    _set_new_state(game_state)


def _set_new_state(game_state: GameState) -> None:
    if game_state.entity_manager.id_card_active is not None:
        if game_state.effect_queue:
            raise InvalidStateError(
                f"Can't enter {CombatState.AWAIT_TARGET_CARD} state with non-empty effect queue"
            )

        game_state.combat_state = CombatState.AWAIT_TARGET_CARD

        return

    if game_state.effect_queue:
        game_state.combat_state = CombatState.AWAIT_TARGET_EFFECT

        return

    game_state.combat_state = CombatState.DEFAULT

    return


def start_combat(game_state: GameState) -> None:
    # Queue start of combat effects
    effects = [Effect(EffectType.COMBAT_START)]
    add_to_bot(game_state.effect_queue, *effects)

    # Process them
    process_effect_queue(game_state.entity_manager, game_state.effect_queue)

    # Set new state
    _set_new_state(game_state)


def main(game_state: GameState, select_action_fn: Callable[[CombatView], Action]) -> None:
    start_combat(game_state)

    while not is_game_over(game_state.entity_manager):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(game_state)
        draw_combat(combat_view)

        # Get action from agent
        action, _ = select_action_fn(combat_view)

        # Game step
        step(game_state, action)

    # TODO: combat end


if __name__ == "__main__":
    from src.game.combat.create import create_combat_state
    from src.rl.policies import PolicyRandom

    game_state = create_combat_state()

    main(game_state, PolicyRandom().select_action)
