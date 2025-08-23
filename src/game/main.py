import os
from dataclasses import replace
from typing import Callable

from src.game.action import Action
from src.game.action import ActionType
from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.draw import get_action_str
from src.game.draw import get_view_game_state_str
from src.game.engine.effect_queue import add_to_bot
from src.game.engine.effect_queue import add_to_top
from src.game.engine.effect_queue import process_effect_queue
from src.game.map_ import RoomType
from src.game.state import GameState
from src.game.utils import does_card_require_target
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.policies import PolicyRandom
from src.rl.policies import SelectActionMetadata


_REST_SITE_REST_HEALTH_GAIN_FACTOR = 0.30
_NCOL, _ = os.get_terminal_size()


class InvalidActionError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _handle_combat_monster_select(
    game_state: GameState, index: int
) -> tuple[list[Effect], list[Effect]]:
    id_monster = game_state.entity_manager.id_monsters[index]

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        # Queue is empty in this state. First the card's target is set, then the active card is
        # cleared, then the card is played (its effects are added to the top of the queue), and
        # finally the card's target is cleared
        return (
            [
                Effect(EffectType.TARGET_CARD_SET, id_target=id_monster),
                Effect(EffectType.CARD_ACTIVE_CLEAR),
                Effect(EffectType.CARD_PLAY, id_target=game_state.entity_manager.id_card_active),
                Effect(EffectType.TARGET_CARD_CLEAR),
            ],
            [],
        )

    raise ValueError(f"Can't select a monster on state {game_state.fsm}")


def _handle_combat_card_in_hand_select(
    game_state: GameState, index: int
) -> tuple[list[Effect], list[Effect]]:
    # Get the selected card's id
    id_card = game_state.entity_manager.id_cards_in_hand[index]

    if game_state.fsm == FSM.COMBAT_DEFAULT:
        card = game_state.entity_manager.entities[id_card]

        # Check if there's enough energy to play it
        energy_current = game_state.entity_manager.entities[
            game_state.entity_manager.id_energy
        ].current
        if card.cost > energy_current:
            raise InvalidActionError(f"Can't select card {card} with {energy_current} energy")

        # If the card requires targeting, set it as active and return
        if does_card_require_target(card):
            return (
                [Effect(EffectType.CARD_ACTIVE_SET, id_target=id_card)],
                [],
            )

        # Else, play it right away. This is different from the original game's implementation,
        # where cards that don't need a target are still set as active and await the player's
        # confirmation
        return (
            [Effect(EffectType.CARD_PLAY, id_target=id_card)],
            [],
        )

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_DISCARD:
        # In this state, the effect at the top of the queue is waiting for a target. We replace
        # `id_target` in the effect and return no other effects
        game_state.effect_queue[0] = replace(game_state.effect_queue[0], id_target=id_card)

        return [], []

    raise ValueError(f"Can't select a card in the hand in state {game_state.fsm}")


def _handle_rest_site_rest(game_state: GameState) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.REST_SITE:
        raise InvalidActionError(f"Can't rest on state {game_state.fsm}")

    character = game_state.entity_manager.entities[game_state.entity_manager.id_character]
    health_gain_value = int(_REST_SITE_REST_HEALTH_GAIN_FACTOR * character.health_max)

    map_node_active = game_state.entity_manager.entities[
        game_state.entity_manager.id_map_node_active
    ]
    if map_node_active.y == len(game_state.entity_manager.id_map_nodes) - 1:
        # Boss fight
        return [
            Effect(
                EffectType.HEALTH_GAIN,
                health_gain_value,
                id_target=game_state.entity_manager.id_character,
            ),
            Effect(
                EffectType.MAP_NODE_ACTIVE_SET,
                id_target=game_state.entity_manager.id_map_node_boss,
            ),
        ], []

    return [
        Effect(
            EffectType.HEALTH_GAIN,
            health_gain_value,
            id_target=game_state.entity_manager.id_character,
        ),
        # Add an effect to select the next map node
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        ),
    ], []


def _handle_rest_site_upgrade(
    game_state: GameState, index: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.REST_SITE:
        raise InvalidActionError(f"Can't upgrade on state {game_state.fsm}")

    id_card = game_state.entity_manager.id_cards_in_deck[index]
    card = game_state.entity_manager.entities[id_card]
    if card.name.endswith("+"):
        raise InvalidActionError("Can't upgrade an already-upgraded card")

    map_node_active = game_state.entity_manager.entities[
        game_state.entity_manager.id_map_node_active
    ]
    if map_node_active.y == len(game_state.entity_manager.id_map_nodes) - 1:
        # Boss fight
        return [
            Effect(EffectType.CARD_UPGRADE, id_target=id_card),
            Effect(
                EffectType.MAP_NODE_ACTIVE_SET,
                id_target=game_state.entity_manager.id_map_node_boss,
            ),
        ], []

    return [
        Effect(EffectType.CARD_UPGRADE, id_target=id_card),
        # Add an effect to select the next map node
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        ),
    ], []


def _handle_map_node_select(
    game_state: GameState, index: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.MAP:
        raise InvalidActionError(f"Can't select a map node on state {game_state.fsm}")

    if game_state.entity_manager.id_map_node_active is None:
        # Starting node
        y_next = 0
        x_valid = [
            x
            for x, node in enumerate(game_state.entity_manager.id_map_nodes[0])
            if node is not None
        ]

    else:
        # Intermediate node
        map_node_active = game_state.entity_manager.entities[
            game_state.entity_manager.id_map_node_active
        ]
        y_next = map_node_active.y + 1
        x_valid = map_node_active.x_next

    if index not in x_valid:
        raise InvalidActionError(
            f"Can't select node on x = {index} on y = {y_next}. Valid options: {x_valid}"
        )

    # Get the id of the selected node, replace it in the top effect (saved before clearing the
    # queue), and append it to the queue
    id_map_node = game_state.entity_manager.id_map_nodes[y_next][index]
    game_state.effect_queue[0] = replace(game_state.effect_queue[0], id_target=id_map_node)

    return [], []


def _handle_card_reward_select(
    game_state: GameState, index: int
) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.CARD_REWARD:
        raise InvalidActionError("TODO: add message")

    effect = game_state.effect_queue[0]
    id_target = game_state.entity_manager.id_card_reward[index]
    game_state.effect_queue[0] = replace(effect, id_target=id_target)

    return [
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        )
    ], []


def _handle_card_reward_skip(game_state: GameState) -> tuple[list[Effect], list[Effect]]:
    if game_state.fsm != FSM.CARD_REWARD:
        raise InvalidActionError("TODO: add message")

    game_state.effect_queue.popleft()

    return [
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        )
    ], []


def handle_action(game_state: GameState, action: Action) -> tuple[list[Effect], list[Effect]]:
    if action.type == ActionType.CARD_REWARD_SELECT:
        return _handle_card_reward_select(game_state, action.index)

    if action.type == ActionType.CARD_REWARD_SKIP:
        return _handle_card_reward_skip(game_state)

    if action.type == ActionType.COMBAT_CARD_IN_HAND_SELECT:
        return _handle_combat_card_in_hand_select(game_state, action.index)

    if action.type == ActionType.COMBAT_MONSTER_SELECT:
        return _handle_combat_monster_select(game_state, action.index)

    if action.type == ActionType.COMBAT_TURN_END:
        return (
            [Effect(EffectType.TURN_END, id_target=game_state.entity_manager.id_character)],
            [],
        )

    if action.type == ActionType.REST_SITE_REST:
        return _handle_rest_site_rest(game_state)

    if action.type == ActionType.REST_SITE_UPGRADE:
        return _handle_rest_site_upgrade(game_state, action.index)

    if action.type == ActionType.MAP_NODE_SELECT:
        return _handle_map_node_select(game_state, action.index)

    raise InvalidActionError(f"Unsupported action type: {action.type}")


def step(game_state: GameState, action: Action) -> None:
    # Handle action
    effects_bot, effects_top = handle_action(game_state, action)

    # Add new effects to the queue
    add_to_bot(game_state.effect_queue, *effects_bot)
    add_to_top(game_state.effect_queue, *effects_top)

    # Process round
    process_effect_queue(game_state)

    # Set new state
    game_state.fsm = _get_new_fsm(game_state)


def _get_new_fsm(game_state: GameState) -> FSM:
    if game_state.effect_queue:
        effect_top = game_state.effect_queue[0]

        if effect_top.type == EffectType.GAME_END:
            return FSM.GAME_OVER

        if effect_top.type == EffectType.CARD_DISCARD:
            return FSM.COMBAT_AWAIT_TARGET_DISCARD

        if effect_top.type == EffectType.MAP_NODE_ACTIVE_SET:
            return FSM.MAP

        if effect_top.type == EffectType.CARD_REWARD_SELECT:
            return FSM.CARD_REWARD

        raise ValueError(f"Unsupported pending effect type: {effect_top.type}")

    if game_state.entity_manager.id_card_active is not None:
        return FSM.COMBAT_AWAIT_TARGET_CARD

    # At this point, the effect queue is clear and there's no active card. The state the game's in
    # depends on the current room type
    map_node_active = game_state.entity_manager.entities[
        game_state.entity_manager.id_map_node_active
    ]
    if map_node_active.room_type == RoomType.REST_SITE:
        return FSM.REST_SITE

    if (
        map_node_active.room_type == RoomType.COMBAT_MONSTER
        or map_node_active.room_type == RoomType.COMBAT_BOSS
    ):
        return FSM.COMBAT_DEFAULT

    raise ValueError(f"Unsupported room type: {map_node_active.room_type}")


def initialize_game_state(game_state: GameState) -> None:
    # Kick-off game with an effect to select the starting map node
    add_to_bot(
        game_state.effect_queue,
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        ),
    )

    # Set new state
    game_state.fsm = _get_new_fsm(game_state)


def main(
    game_state: GameState,
    select_action_fn: Callable[[ViewGameState], tuple[Action, SelectActionMetadata]],
    draw: bool = False,
) -> GameState:
    initialize_game_state(game_state)

    # Loop
    while game_state.fsm != FSM.GAME_OVER:
        # Get game state view and draw it on the terminal
        game_state_view = get_view_game_state(game_state)

        # Get action from agent
        action, _ = select_action_fn(game_state_view)

        # Game step
        step(game_state, action)

        # Draw on terminal
        if draw:
            view_game_state_str = get_view_game_state_str(game_state_view)
            action_str = get_action_str(action, game_state_view)
            print(view_game_state_str)
            print("-" * _NCOL)
            print(action_str)
            print("-" * _NCOL)

    return game_state


if __name__ == "__main__":
    ascension_level = 1

    game_state = create_game_state(ascension_level)
    main(game_state, PolicyRandom().select_action, draw=True)
