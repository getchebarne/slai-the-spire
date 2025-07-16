import os
import sys
import termios
import tty
from copy import deepcopy

from pynput import keyboard as kb

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


_MARGIN_X = 2


# Intialize a container to store the pressed key
cont_key: list[kb.Key | None] = [None]
cont_idx_hover: list[int] = [0, 0]


def _disable_echo_and_buffering() -> None:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    return old_settings


def _on_press(key: kb.Key) -> None:
    if key == kb.Key.right:
        cont_idx_hover[0] += 1
    elif key == kb.Key.left:
        cont_idx_hover[0] -= 1
    elif key == kb.Key.up:
        cont_idx_hover[1] -= 1
    elif key == kb.Key.down:
        cont_idx_hover[1] += 1

    cont_key[0] = key

    return


def _get_grid_main(height: int, width: int) -> Grid:
    grid = init_grid(height, width)

    # Border
    grid = put_border(grid)

    # Title
    title = " Slai The Spire "
    grid = put_str(grid, title, y=0, x=_MARGIN_X)

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

    _disable_echo_and_buffering()

    # Start game
    game_state = create_game_state(ascension_level=1)
    game_state.fsm = _get_new_fsm(game_state)

    # Main loop that prints the last key pressed
    print("\033[?25l", end="")  # Hides the cursor
    card_upgrade = False
    while game_state.fsm != FSM.GAME_OVER:
        cont_key[0] = None
        grid_main = _get_grid_main(HEIGHT_MAIN, WIDTH_MAIN)

        if game_state.fsm == FSM.MAP:
            grid_map = get_grid_map(
                game_state.entity_manager.map_nodes,
                game_state.entity_manager.map_node_active,
                idx_hover=cont_idx_hover[0],
            )
            grid_main = paste_grid(
                grid_map,
                grid_main,
                y=(HEIGHT_MAIN - len(grid_map)) // 2,
                x=(WIDTH_MAIN - len(grid_map[0])) // 2,
            )
        elif (
            game_state.fsm == FSM.COMBAT_DEFAULT
            or game_state.fsm == FSM.COMBAT_AWAIT_TARGET_DISCARD
            or game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD
        ):
            grid_combat = get_grid_combat(
                HEIGHT_MAIN - 2, WIDTH_MAIN - 2, game_state, cont_idx_hover[0]
            )
            grid_main = paste_grid(
                grid_combat,
                grid_main,
                y=(HEIGHT_MAIN - len(grid_combat)) // 2,
                x=(WIDTH_MAIN - len(grid_combat[0])) // 2,
            )

        elif game_state.fsm == FSM.CARD_REWARD:
            grid_combat_reward = get_grid_combat_reward(game_state, cont_idx_hover[0])
            grid_main = paste_grid(
                grid_combat_reward,
                grid_main,
                y=(HEIGHT_MAIN - len(grid_combat_reward)) // 2,
                x=(WIDTH_MAIN - len(grid_combat_reward[0])) // 2,
            )

        elif game_state.fsm == FSM.REST_SITE:
            grid_rest_site = get_grid_rest_site(game_state, cont_idx_hover[0], card_upgrade)
            grid_main = paste_grid(
                grid_rest_site,
                grid_main,
                y=(HEIGHT_MAIN - len(grid_rest_site)) // 2,
                x=(WIDTH_MAIN - len(grid_rest_site[0])) // 2,
            )

        else:
            raise ValueError(game_state.fsm)

        # Print
        print_grid(grid_main)

        # Block until player input is received
        while cont_key[0] is None:
            pass

        # TODO: improve this code it's ugly
        key = cont_key[0]
        if key == kb.Key.enter:
            if game_state.fsm == FSM.REST_SITE and not card_upgrade:
                if cont_idx_hover[0] == 0:
                    action = Action(ActionType.REST_SITE_REST)
                else:
                    card_upgrade = True
                    continue

            action = _get_action(game_state, cont_idx_hover[0], card_upgrade)
            step(game_state, action)

            card_upgrade = False
            cont_idx_hover[0] = 0

        elif hasattr(key, "char"):
            if key.char == "e":
                action = Action(ActionType.COMBAT_TURN_END)
                step(game_state, action)

            elif key.char == "d":
                # Show card description
                card = game_state.entity_manager.cards_in_hand[cont_idx_hover[0]]
                grid = get_grid_card_zoomed(card, 20, 28)
                grid_main_prev = deepcopy(grid_main)
                grid_main = paste_grid(
                    grid,
                    grid_main,
                    y=(len(grid_main) - len(grid)) // 2 - 1,
                    x=(len(grid_main[0]) - len(grid[0])) // 2,
                )
                print_grid(grid_main)
                while not hasattr(cont_key[0], "char") or cont_key[0].char != "b":
                    pass

                grid_main = grid_main_prev

            elif key.char == "k":
                # Show deck
                grid_main_prev = deepcopy(grid_main)
                while not hasattr(cont_key[0], "char") or cont_key[0].char != "b":
                    os.system("cls" if os.name == "nt" else "clear")
                    grid = get_grid_card_paginate(
                        game_state.entity_manager.cards_in_deck,
                        3,
                        3,
                        height_card=10,
                        width_card=14,
                        y_gap=1,
                        offset=cont_idx_hover[1],
                    )
                    grid_main = paste_grid(
                        grid,
                        grid_main,
                        y=(len(grid_main) - len(grid)) // 2 - 1,
                        x=(len(grid_main[0]) - len(grid[0])) // 2,
                    )
                    print_grid(grid_main)
                    while cont_key[0] != kb.Key.down and cont_key[0] != kb.Key.up:
                        pass

                    cont_key[0] = None

                grid_main = grid_main_prev

        # Correct index according to the game's state
        cont_idx_hover[0] = _correct_index(cont_idx_hover[0], game_state, card_upgrade)


if __name__ == "__main__":
    main()
