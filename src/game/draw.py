import os
import re

from src.game.action import Action
from src.game.action import ActionType
from src.game.entity.map_node import RoomType
from src.game.view.actor import ViewActor
from src.game.view.card import ViewCard
from src.game.view.energy import ViewEnergy
from src.game.view.fsm import ViewFSM
from src.game.view.map_ import ViewMap
from src.game.view.monster import Intent
from src.game.view.monster import ViewMonster
from src.game.view.state import ViewGameState


# Get the terminal width
N_COL, _ = os.get_terminal_size()

# For calculating the actual length of strings with ANSI escape characters
ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

# Colors
RED = "\033[31;1m"
CYAN = "\033[36;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"
WHITE = "\033[37;1m"


def _energy_str(energy: ViewEnergy) -> str:
    return f"ENERGY: {energy.current}/{energy.max}"


def _card_str(card: ViewCard) -> str:
    return f"({card.cost}) {card.name}"


def _hand_str(hand: list[ViewCard]) -> str:
    card_strings = []
    for card_view in hand:
        if card_view.is_active:
            card_strings.append(f"{GREEN}{_card_str(card_view)}{RESET}")

            continue

        card_strings.append(f"{_card_str(card_view)}")

    return f"HAND: {' / '.join(card_strings)}"


def _health_str(health_current: int, health_max: int) -> str:
    return f"HP: {health_current}/{health_max}"


def _block_str(block_current: int) -> str:
    return f"BLK: {block_current}"


def _actor_str(actor: ViewActor, n_col: int = 0) -> str:
    modifier_strs = "\n".join(
        [
            f"{modifier_view_type.name}: {stacks_current}"
            for modifier_view_type, stacks_current in actor.modifiers.items()
        ]
    )

    return (
        f"{WHITE}{actor.name:>{n_col}}{RESET}\n"
        f"{WHITE}{'-' * len(actor.name):>{n_col}}{RESET}\n"
        f"{RED}{_health_str(actor.health_current, actor.health_max):>{n_col}}{RESET}\n"
        f"{CYAN}{_block_str(actor.block_current):>{n_col}}{RESET}\n"
        f"{modifier_strs:>{n_col}}"
    )


def _intent_str(intent: Intent | None) -> str:
    str_ = ""
    if intent is None:
        return str_

    if intent.damage is not None:
        str_ = f"{str_}{intent.damage} x {intent.instances}"

    if intent.block:
        if str_ != "":
            str_ = f"{str_} & Blocking"

        else:
            str_ = "Blocking"

    if intent.buff:
        if str_ != "":
            str_ = f"{str_} & Buffing"

        else:
            str_ = "Buffing"

    return str_


def _monster_str(monster_view: ViewMonster) -> str:
    # Get base actor string
    str_ = _actor_str(monster_view, N_COL)

    # Split into lines
    lines = str_.split("\n")

    # Insert monster's intent at first position
    lines.insert(0, _intent_str(monster_view.intent))

    # Align lines to the right of the terminal
    right_aligned_lines = [f"{line:>{N_COL}}" for line in lines]

    # Stitch together and return
    return "\n".join(right_aligned_lines)


# TODO: improve
def _map_str(map_: ViewMap) -> None:
    map_height = len(map_.nodes)
    map_width = len(map_.nodes[0])
    grid_rows = map_height * 2
    grid_cols = map_width * 3
    grid = [[" " for _ in range(grid_cols + 3)] for _ in range(grid_rows)]

    # Place only active nodes
    for y in range(map_height):
        if y == map_.y_current:
            grid[2 * y][0:3] = ">>>"

        for x in range(map_width):
            # Place node
            map_node = map_.nodes[y][x]
            if map_node is None:
                continue

            gx = x * 3 + 4  # center column
            gy = y * 2  # top row of the 2-row cell
            if map_node.room_type == RoomType.COMBAT_MONSTER:
                char = "M"
            else:
                char = "R"

            if y == map_.y_current and x == map_.x_current:
                char = f"{GREEN}{char}{RESET}"

            grid[gy][gx] = char

            # Place edges
            edge_row = gy + 1
            for x_next in map_node.x_next:
                dx = x_next - x

                if dx == -1:
                    grid[edge_row][gx - 1] = "\\"
                elif dx == 0:
                    grid[edge_row][gx] = "|"
                elif dx == 1:
                    grid[edge_row][gx + 1] = "/"

    return "\n".join(reversed(["".join(row) for row in grid]))


def _get_visible_length(text: str) -> int:
    return len(ANSI_ESCAPE_RE.sub("", text))


def _center_text(text: str) -> str:
    padding = (N_COL - _get_visible_length(text)) // 2
    text = " " * max(padding, 0) + text

    return text


def _rest_site_str() -> str:
    return """
(
)
(  (
)
( ) (
) /\\
( / | (`'(
_ -.;_/ \\--._
(_;-// | \\ \\-'.\\
( `.__ _  ___,')
`'(_ )_)(_)_)'
"""


def get_view_game_state_str(view_game_state: ViewGameState) -> str:
    str_fsm = view_game_state.fsm.name.replace("_", " ")
    str_fsm_underline = "¯" * len(str_fsm)
    str_fsm = f"{_center_text(str_fsm)}\n{_center_text(str_fsm_underline)}"

    if view_game_state.fsm == ViewFSM.REST_SITE:
        str_ = "\n".join(_center_text(line) for line in _rest_site_str().split("\n"))

    elif view_game_state.fsm == ViewFSM.MAP:
        str_ = "\n".join(_center_text(line) for line in _map_str(view_game_state.map).split("\n"))

    elif view_game_state.fsm == ViewFSM.CARD_REWARD:
        str_ = "\n".join(
            [_center_text(_card_str(view_card)) for view_card in view_game_state.reward_combat]
        )

    else:
        # Monsters
        monster_strs = "\n\n".join(
            [f"{_monster_str(monster)}" for monster in view_game_state.monsters]
        )

        # Character
        character_str = _actor_str(view_game_state.character)

        # Energy
        energy_str = _energy_str(view_game_state.energy)

        # Hand
        hand_str = _hand_str(view_game_state.hand)

        # All
        str_ = f"{monster_strs}\n{character_str}\n{energy_str}\n{hand_str}"

    str_ = f"{_center_text(str_fsm)}\n{str_}"

    return str_


def get_action_str(action: Action, view_game_state: ViewGameState) -> str:
    if action.type == ActionType.CARD_REWARD_SELECT:
        return f"> Add {view_game_state.reward_combat[action.index].name} to the Deck!"

    if action.type == ActionType.CARD_REWARD_SKIP:
        return "> Skip card rewards."

    if action.type == ActionType.COMBAT_CARD_IN_HAND_SELECT:
        card = view_game_state.hand[action.index]
        if view_game_state.fsm == ViewFSM.COMBAT_DEFAULT:
            if card.requires_target:
                return f"> Select {card.name}."

            return f"> Play {card.name}."

        if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
            return f"> Discard {card.name}."

        raise ValueError(f"Unsupported FSM state: {view_game_state.fsm}")

    if action.type == ActionType.COMBAT_MONSTER_SELECT:
        return (
            f"> Select {view_game_state.monsters[action.index].name} at position {action.index}."
        )

    if action.type == ActionType.COMBAT_TURN_END:
        return "> End turn."

    if action.type == ActionType.MAP_NODE_SELECT:
        if view_game_state.map.x_current is None:
            return f"> Select map node at x = {action.index}."

        dx = action.index - view_game_state.map.x_current
        if dx == 1:
            direction = "right"

        elif dx == -1:
            direction = "left"

        else:
            direction = "middle"

        return f"> Select map node to the {direction}."

    if action.type == ActionType.REST_SITE_REST:
        return "> Rest."

    if action.type == ActionType.REST_SITE_UPGRADE:
        return f"> Upgrade {view_game_state.deck[action.index].name}."
