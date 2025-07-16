import os
import sys
import termios
import tty
from enum import Enum

from pynput import keyboard as kb
from pynput.keyboard import Key

from src.game.action import Action
from src.game.action import ActionType
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.draw_3.card import get_grid_card_paginate
from src.game.draw_3.card import get_grid_card_zoomed
from src.game.draw_3.combat import get_grid_combat
from src.game.draw_3.combat_reward import get_grid_combat_reward
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import paste_grid
from src.game.draw_3.grid import print_grid
from src.game.draw_3.grid import put_border
from src.game.draw_3.grid import put_str
from src.game.draw_3.layout import HEIGHT_MAIN
from src.game.draw_3.layout import WIDTH_MAIN
from src.game.draw_3.map import get_grid_map
from src.game.draw_3.rest_site import get_grid_rest_site
from src.game.main import _get_new_fsm
from src.game.main import step
from src.game.state import GameState


class WindowType(Enum):
    COMBAT = "COMBAT"
    MAP = "MAP"
    VIEW_CARD = "VIEW_CARD"
    VIEW_DECK = "VIEW_DECK"


# Intialize a container to store the pressed key
cont_key: list[Key | None] = [None]
cont_idx_hover: list[int] = [0, 0]


def _disable_echo_and_buffering() -> None:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    return old_settings


def _on_press(key: Key) -> None:
    if key == Key.right:
        cont_idx_hover[0] += 1
    elif key == Key.left:
        cont_idx_hover[0] -= 1
    elif key == Key.up:
        cont_idx_hover[1] -= 1
    elif key == Key.down:
        cont_idx_hover[1] += 1

    cont_key[0] = key

    return


def _get_grid_main(height: int, width: int) -> Grid:
    grid = init_grid(height, width)

    # Border
    grid = put_border(grid)

    # Title
    title = " Slai The Spire "
    grid = put_str(grid, title, y=0, x=2)

    return grid


def _correct_index(idx_hover: int, game_state: GameState, card_upgrade: bool) -> int:
    if game_state.fsm == FSM.MAP:
        if game_state.entity_manager.map_node_active is None:
            return idx_hover % len(
                [
                    x
                    for x, node in enumerate(game_state.entity_manager.map_nodes[0])
                    if node is not None
                ]
            )

        return idx_hover % len(game_state.entity_manager.map_node_active.x_next)

    if game_state.fsm == FSM.COMBAT_DEFAULT or game_state.fsm == FSM.COMBAT_AWAIT_TARGET_DISCARD:
        return idx_hover % len(game_state.entity_manager.cards_in_hand)

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        return idx_hover % len(game_state.entity_manager.monsters)

    if game_state.fsm == FSM.CARD_REWARD:
        return idx_hover % len(game_state.entity_manager.cards_reward)

    if game_state.fsm == FSM.REST_SITE:
        if card_upgrade:
            return idx_hover % len(game_state.entity_manager.cards_in_deck)

        return idx_hover % 2

    raise ValueError(game_state.fsm)


def _get_action(game_state: GameState, idx_hover: int, card_upgrade: bool) -> Action:
    if game_state.fsm == FSM.MAP:
        if game_state.entity_manager.map_node_active is None:
            x_next = [
                x
                for x, node in enumerate(game_state.entity_manager.map_nodes[0])
                if node is not None
            ]
        else:
            x_next = sorted(list(game_state.entity_manager.map_node_active.x_next))

        return Action(ActionType.MAP_NODE_SELECT, index=x_next[idx_hover])

    if game_state.fsm == FSM.COMBAT_DEFAULT or game_state.fsm == FSM.COMBAT_AWAIT_TARGET_DISCARD:
        return Action(
            ActionType.COMBAT_CARD_IN_HAND_SELECT,
            list(range(len(game_state.entity_manager.id_cards_in_hand)))[idx_hover],
        )

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        return Action(
            ActionType.COMBAT_MONSTER_SELECT,
            list(range(len(game_state.entity_manager.id_monsters)))[idx_hover],
        )

    if game_state.fsm == FSM.CARD_REWARD:
        return Action(
            ActionType.CARD_REWARD_SELECT,
            list(range(len(game_state.entity_manager.cards_reward)))[idx_hover],
        )

    if game_state.fsm == FSM.REST_SITE:
        return Action(
            ActionType.REST_SITE_UPGRADE,
            list(range(len(game_state.entity_manager.cards_in_deck)))[idx_hover],
        )

    raise ValueError(game_state.fsm)


def main() -> None:
    # Start the keyboard listener in a separate thread
    listener = kb.Listener(on_press=_on_press)
    listener.start()

    # Disable echo and buffering, hide cursor
    _disable_echo_and_buffering()
    print("\033[?25l", end="")

    # Start game
    game_state = create_game_state(ascension_level=1)
    game_state.fsm = _get_new_fsm(game_state)

    # Main loop
    card_upgrade = False
    window_stack = [None]
    while game_state.fsm != FSM.GAME_OVER:
        # Reset key press container
        cont_key[0] = None

        match game_state.fsm:
            case (
                FSM.COMBAT_DEFAULT | FSM.COMBAT_AWAIT_TARGET_CARD | FSM.COMBAT_AWAIT_TARGET_DISCARD
            ):
                window_stack[0] = WindowType.COMBAT

            case FSM.MAP:
                window_stack[0] = WindowType.MAP

            case _:
                raise ValueError(game_state.fsm)

        # Print
        grid = _get_grid_main(HEIGHT_MAIN, WIDTH_MAIN)
        for window_type in window_stack:
            match window_type:
                case WindowType.COMBAT:
                    grid_paste = get_grid_combat(
                        HEIGHT_MAIN - 2, WIDTH_MAIN - 2, game_state, cont_idx_hover[0]
                    )

                case WindowType.MAP:
                    grid_paste = get_grid_map(
                        game_state.entity_manager.map_nodes,
                        game_state.entity_manager.map_node_active,
                        idx_hover=cont_idx_hover[0],
                    )

                case WindowType.VIEW_CARD:
                    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
                        card = game_state.entity_manager.card_active
                    else:
                        card = game_state.entity_manager.cards_in_hand[cont_idx_hover[0]]

                    grid_paste = get_grid_card_zoomed(card, 20, 28)

                case WindowType.VIEW_DECK:
                    grid_paste = get_grid_card_paginate(
                        game_state.entity_manager.cards_in_deck,
                        num_rows=3,
                        num_cols=3,
                        height_card=10,
                        width_card=14,
                        y_gap=1,
                        offset_y=cont_idx_hover[1],
                        offset_x=cont_idx_hover[0],
                    )

                case _:
                    raise ValueError(window_type)

            grid = paste_grid(
                grid_paste,
                grid,
                y=(HEIGHT_MAIN - len(grid_paste)) // 2,
                x=(WIDTH_MAIN - len(grid_paste[0])) // 2,
            )

        os.system("clear")
        print_grid(grid)

        # Block until input is received
        while cont_key[0] is None:
            pass

        os.system("clear")

        # TODO: improve this code it's ugly
        key = cont_key[0]
        action = None
        if hasattr(key, "char"):
            match key.char:
                case "e":
                    if window_stack[-1] == WindowType.COMBAT:
                        action = Action(ActionType.COMBAT_TURN_END)

                case "v":
                    window_stack.append(WindowType.VIEW_CARD)

                case "k":
                    window_stack = window_stack[:2]
                    window_stack.append(WindowType.VIEW_DECK)

                case "b":
                    if len(window_stack) > 1:
                        window_stack.pop()

        elif key == Key.esc:
            window_stack = window_stack[:1]

        elif key == Key.enter:
            action = _get_action(game_state, cont_idx_hover[0], card_upgrade)
            cont_idx_hover[0] = 0

        if action is not None:
            step(game_state, action)

        # Correct index according to the game's state
        cont_idx_hover[0] = _correct_index(cont_idx_hover[0], game_state, card_upgrade)


if __name__ == "__main__":
    main()
