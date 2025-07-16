import curses
import os

from src.game.action import Action
from src.game.action import ActionType
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.draw_2.actor import _WIDTH_ACTOR
from src.game.draw_2.actor import draw_actor
from src.game.draw_2.actor import get_window_actor_height
from src.game.draw_2.card import draw_card
from src.game.draw_2.color import SELECTED
from src.game.draw_2.color import init_colors
from src.game.entity.map_node import RoomType
from src.game.main import _get_new_fsm
from src.game.main import step
from src.game.state import GameState
from src.game.view.fsm import ViewFSM
from src.game.view.map_ import ViewMap
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state


_HEIGHT = 40
_WIDTH = 120
_HEIGHT_CARD = 10
_HEIGHT_ENERGY = 3
_WIDTH_CARD = 14
_WIDTH_ENERGY = 5
_GAP_CARD = 1
_GAP_MONST = 1
_MARGIN_X = 2
_MARGIN_Y = 2


def _draw_fsm_combat(view_game_state: ViewGameState, index_selected: int) -> None:
    # Cards
    y_begin = _HEIGHT - _HEIGHT_CARD - 1
    for idx, card_view in enumerate(view_game_state.hand):
        x_begin = _MARGIN_X + idx * (_WIDTH_CARD + _GAP_CARD)
        color_pair = (
            0
            if idx != index_selected or view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_CARD
            else SELECTED
        )
        draw_card(card_view, y_begin, x_begin, color_pair)

    # Character
    y_char = y_begin - get_window_actor_height(view_game_state.character)
    win_char = draw_actor(view_game_state.character, y_char, _MARGIN_X)
    win_char.refresh()

    # Energy
    y_energy = y_char - _HEIGHT_ENERGY
    win_energy = curses.newwin(_HEIGHT_ENERGY, _WIDTH_ENERGY, y_energy, _MARGIN_X)
    win_energy.border()
    win_energy.addstr(1, 1, f"{view_game_state.energy.current}/{view_game_state.energy.max}")
    win_energy.refresh()

    # Monsters
    y_monster = 1
    num_monsters = len(view_game_state.monsters)
    for idx, monster in enumerate(reversed(view_game_state.monsters)):
        x_begin = _WIDTH - _MARGIN_X - _WIDTH_ACTOR - idx * (_WIDTH_ACTOR + _GAP_MONST)
        monster_index = num_monsters - 1 - idx  # maps reversed index to original index
        color_pair = (
            0
            if monster_index != index_selected
            or view_game_state.fsm != ViewFSM.COMBAT_AWAIT_TARGET_CARD
            else SELECTED
        )
        win_monster = draw_actor(monster, y_monster, x_begin, color_pair)
        win_monster.refresh()


def _draw_map(map_: ViewMap, index_selected: int) -> None:
    map_height = 2 * len(map_.nodes)
    map_width = 3 * len(map_.nodes[0])
    y_next = (
        len(map_.nodes) - 1 if map_.y_current is None else len(map_.nodes) - map_.y_current - 1
    )
    xs = [x for x, node in enumerate(list(reversed(map_.nodes))[y_next]) if node is not None]
    index_selected = index_selected % len(xs)

    win_map = curses.newwin(
        map_height, map_width, (_HEIGHT - map_height) // 2, (_WIDTH - map_width) // 2
    )

    for y, row in enumerate(reversed(map_.nodes)):
        for x, node in enumerate(row):
            if node is None:
                continue

            # Node
            if node.room_type == RoomType.COMBAT_MONSTER:
                ch = "M"
            elif node.room_type == RoomType.REST_SITE:
                ch = "R"
            else:
                raise ValueError

            if y == y_next:
                if x == xs[index_selected]:
                    win_map.addch(1 + 2 * y, 1 + 3 * x, ch, curses.color_pair(SELECTED))
                else:
                    win_map.addch(1 + 2 * y, 1 + 3 * x, ch)
            else:
                win_map.addch(1 + 2 * y, 1 + 3 * x, ch)

            # Edges
            for x_next in node.x_next:
                dx = x_next - x
                if dx == -1:
                    ch = "\\"
                    win_map.addch(2 * y, 3 * x, ch)

                elif dx == 0:
                    ch = "|"
                    win_map.addch(
                        2 * y,
                        3 * x + 1,
                        ch,
                    )
                else:
                    ch = "/"
                    win_map.addch(2 * y, 3 * x + 2, ch)

    win_map.refresh()


def _handle_press_enter(game_state: GameState, index_selected: int) -> Action:
    if game_state.fsm == FSM.MAP:
        if game_state.entity_manager.map_node_active is None:
            x = [
                x
                for x, node in enumerate(game_state.entity_manager.map_nodes[0])
                if node is not None
            ]
        else:
            raise ValueError

        return Action(ActionType.MAP_NODE_SELECT, index=x[index_selected])

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        return Action(
            ActionType.COMBAT_MONSTER_SELECT,
            list(range(len(game_state.entity_manager.id_monsters)))[index_selected],
        )

    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_DISCARD:
        return Action(
            ActionType.COMBAT_CARD_IN_HAND_SELECT,
            list(range(len(game_state.entity_manager.id_cards_in_hand)))[index_selected],
        )

    if game_state.fsm == FSM.COMBAT_DEFAULT:
        return Action(
            ActionType.COMBAT_CARD_IN_HAND_SELECT,
            list(range(len(game_state.entity_manager.id_cards_in_hand)))[index_selected],
        )

    raise ValueError(game_state.fsm)


def _update_index(index: int, view_game_state: ViewGameState) -> int:
    if view_game_state.fsm == ViewFSM.MAP:
        if view_game_state.map.y_current is None:
            y_next = 0
        else:
            y_next = view_game_state.map.y_current + 1

        return index % len(
            [x for x, node in enumerate(view_game_state.map.nodes[y_next]) if node is not None]
        )

    if view_game_state.fsm == ViewFSM.COMBAT_DEFAULT:
        return index % len(view_game_state.hand)

    if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_CARD:
        return index % len(view_game_state.monsters)

    if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
        return index % len(view_game_state.hand)

    if view_game_state.fsm == ViewFSM.CARD_REWARD:
        return index % len(view_game_state.reward_combat)

    raise ValueError


def _draw_fsm_combat_reward(view_game_state: ViewGameState, index_selected: int) -> None:
    num_rewards = len(view_game_state.reward_combat)

    y_begin = (_HEIGHT - _HEIGHT_CARD) // 2
    x_start = (_WIDTH - num_rewards * (_WIDTH_CARD * _GAP_CARD)) // 2
    for idx, card_view in enumerate(view_game_state.reward_combat):
        color_pair = 0 if idx != index_selected else SELECTED
        draw_card(card_view, y_begin, x_start, color_pair)
        x_start += _WIDTH_CARD + _GAP_CARD

    return


def main(stdscr):
    curses.curs_set(0)
    init_colors()

    stdscr.clear()
    stdscr.refresh()

    max_y, max_x = stdscr.getmaxyx()

    if max_y < _HEIGHT or max_x < _WIDTH:
        n_col, n_line = os.get_terminal_size()
        stdscr.addstr(
            0,
            0,
            f"Please resize your terminal. Current size: {n_line} x {n_col}. Minimum: {_HEIGHT} x {_WIDTH})",
        )
        stdscr.refresh()
        stdscr.getch()
        return

    game_state = create_game_state(1)
    game_state.fsm = _get_new_fsm(game_state)

    index_selected = 0
    while True:
        view_game_state = get_view_game_state(game_state)

        # Main window
        win_main = curses.newwin(_HEIGHT, _WIDTH, 0, 0)
        win_main.border()
        win_main.addstr(0, 2, "  Slai the Spire  ")
        win_main.refresh()

        if view_game_state.fsm == ViewFSM.MAP:
            _draw_map(view_game_state.map, index_selected)
        elif view_game_state.fsm in (
            ViewFSM.COMBAT_DEFAULT,
            ViewFSM.COMBAT_AWAIT_TARGET_CARD,
            ViewFSM.COMBAT_AWAIT_TARGET_DISCARD,
        ):
            _draw_fsm_combat(view_game_state, index_selected)
        elif view_game_state.fsm == ViewFSM.CARD_REWARD:
            _draw_fsm_combat_reward(view_game_state, index_selected)
        else:
            raise ValueError(view_game_state.fsm)

        key = stdscr.getch()
        if key == curses.KEY_RIGHT:
            index_selected = _update_index(index_selected + 1, view_game_state)
        elif key == curses.KEY_LEFT:
            index_selected = _update_index(index_selected - 1, view_game_state)

        elif key in (10, 13, curses.KEY_ENTER):
            action = _handle_press_enter(game_state, index_selected)
            step(game_state, action)
            index_selected = 0
        elif key == ord("e"):
            # End turn
            action = Action(ActionType.COMBAT_TURN_END)
            step(game_state, action)
            index_selected = 0
        elif key in (ord("q"), 27):  # 'q' or ESC to quit
            break


curses.wrapper(main)
