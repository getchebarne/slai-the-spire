import random
from typing import Callable

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.create import create_game_state
from src.game.combat.drawer import draw_combat
from src.game.combat.utils import does_card_require_target
from src.game.combat.utils import is_character_dead
from src.game.combat.utils import is_combat_over
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.core.fsm import FSM
from src.game.engine.effect_queue import add_to_bot
from src.game.engine.effect_queue import add_to_top
from src.game.engine.effect_queue import process_effect_queue
from src.game.engine.state import GameState
from src.game.level.exordium.combat_cultist import set_level_exoridium_combat_cultist
from src.game.level.exordium.combat_fungi_beast_two import (
    set_level_exoridium_combat_fungi_beast_two,
)
from src.game.level.exordium.combat_jaw_worm import set_level_exoridium_combat_jaw_worm
from src.game.level.rest_site import set_level_rest_site
from src.game.types_ import CombatResult
from src.game.types_ import RoomType
from src.rl.policies import PolicyRandom
from src.rl.policies import SelectActionMetadata


_REST_SITE_REST_HEALTH_GAIN_FACTOR = 0.30


class InvalidActionError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _handle_entity_select(
    game_state: GameState, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm == FSM.COMBAT_DEFAULT:
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

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
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

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_EFFECT:
        # An effect is added at the top of the queue to set the effect's target. The effect that
        # comes inmediately after is going to use this variable to resolve its target. For now,
        # an effect to clear the effect's target is added to the bottom of the queue, but TODO:
        # I think I can't escape adding the clear when the `id_effect_target` is consumed
        return (
            [Effect(EffectType.TARGET_EFFECT_CLEAR, id_target=id_target)],
            [Effect(EffectType.TARGET_EFFECT_SET, id_target=id_target)],
        )

    raise ValueError("TODO: add message")


def _handle_rest_site_rest(game_state: GameState) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.REST_SITE:
        raise InvalidActionError(f"Can't rest on state {game_state.fsm}")

    character = game_state.entity_manager.entities[game_state.entity_manager.id_character]
    health_gain_value = int(_REST_SITE_REST_HEALTH_GAIN_FACTOR * character.health_max)

    return [
        Effect(
            EffectType.HEALTH_GAIN,
            health_gain_value,
            id_target=game_state.entity_manager.id_character,
        )
    ], []


def _handle_rest_site_upgrade(
    game_state: GameState, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.REST_SITE:
        raise InvalidActionError(f"Can't upgrade on state {game_state.fsm}")

    return [Effect(EffectType.CARD_UPGRADE, id_target=id_target)], []


def handle_action(game_state: GameState, action: Action) -> tuple[list[Effect], list[Effect]]:
    if action.type == ActionType.TURN_END:
        return (
            [Effect(EffectType.TURN_END, id_target=game_state.entity_manager.id_character)],
            [],
        )

    if action.type == ActionType.ENTITY_SELECT:
        return _handle_entity_select(game_state, action.target_id)

    if action.type == ActionType.REST_SITE_REST:
        return _handle_rest_site_rest(game_state)

    if action.type == ActionType.REST_SITE_UPGRADE:
        return _handle_rest_site_upgrade(game_state, action.target_id)


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
                f"Can't enter {FSM.COMBAT_AWAIT_TARGET_CARD} state with non-empty effect queue"
            )

        game_state.fsm = FSM.COMBAT_AWAIT_TARGET_CARD

        return

    if game_state.effect_queue:
        game_state.fsm = FSM.COMBAT_AWAIT_TARGET_EFFECT

        return

    game_state.fsm = FSM.COMBAT_DEFAULT

    return


def start_combat(game_state: GameState) -> None:
    # Queue start of combat effect and process it
    add_to_bot(game_state.effect_queue, Effect(EffectType.COMBAT_START))
    process_effect_queue(game_state.entity_manager, game_state.effect_queue)

    # Set new state
    _set_new_state(game_state)


# TODO: `is_combat_over` should be checked after each effect is processed maybe?
def _combat_loop(
    game_state: GameState,
    select_action_fn: Callable[[CombatView], tuple[Action, SelectActionMetadata]],
) -> CombatResult:
    # Set combat TODO: move
    set_level_combat_fn = random.choice(
        [
            set_level_exoridium_combat_cultist,
            set_level_exoridium_combat_jaw_worm,
            set_level_exoridium_combat_fungi_beast_two,
        ]
    )
    set_level_combat_fn(game_state)

    # Start combat
    start_combat(game_state)

    # Loop
    while not is_combat_over(game_state.entity_manager):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(game_state)
        draw_combat(combat_view)

        # Get action from agent
        action, _ = select_action_fn(combat_view)

        # Game step
        step(game_state, action)

    # Combat end
    if is_character_dead(game_state.entity_manager):
        return CombatResult.LOSS

    # Queue end of combat effect and process it
    add_to_bot(game_state.effect_queue, Effect(EffectType.COMBAT_END))
    process_effect_queue(game_state.entity_manager, game_state.effect_queue)

    return CombatResult.WIN


def main(
    game_state: GameState,
    select_action_fn: Callable[[CombatView], tuple[Action, SelectActionMetadata]],
) -> None:
    for room_type in game_state.map_:
        if room_type == RoomType.COMBAT_MONSTER:
            combat_result = _combat_loop(game_state, select_action_fn)

            if combat_result == CombatResult.LOSS:
                break

        if room_type == RoomType.REST_SITE:
            set_level_rest_site(game_state)

            # Get action from agent
            combat_view = view_combat(game_state)
            draw_combat(combat_view)
            action, _ = select_action_fn(combat_view)
            step(game_state, action)
            print(f"{action=}")


if __name__ == "__main__":
    ascension_level = 20
    game_state = create_game_state(ascension_level)
    set_level_exoridium_combat_cultist(game_state)

    main(game_state, PolicyRandom().select_action)
