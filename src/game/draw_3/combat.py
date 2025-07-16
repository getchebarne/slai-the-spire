from src.game.core.fsm import FSM
from src.game.draw_3 import color
from src.game.draw_3 import layout
from src.game.draw_3.actor import get_grid_actor_height
from src.game.draw_3.actor import get_grid_character
from src.game.draw_3.actor import get_grid_monster
from src.game.draw_3.card import get_grid_card
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import paste_grid
from src.game.draw_3.grid import put_border
from src.game.draw_3.grid import put_str
from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.entity.energy import EntityEnergy
from src.game.entity.monster import EntityMonster
from src.game.state import GameState


def get_grid_combat(height: int, width: int, game_state: GameState, idx_hover: int) -> Grid:
    # Intialize grid
    grid_main = init_grid(height, width)

    # Hand
    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        idx_push = game_state.entity_manager.id_cards_in_hand.index(
            game_state.entity_manager.id_card_active
        )
    else:
        idx_push = idx_hover

    grid_hand = _get_grid_hand(
        game_state.entity_manager.cards_in_hand,
        layout.HEIGHT_CARD,
        layout.WIDTH_CARD,
        layout.GAP_CARD,
        layout.COMBAT_CARD_HAND_PUSH,
        idx_push,
    )
    y_hand = height - len(grid_hand)
    grid_main = paste_grid(grid_hand, grid_main, y=y_hand, x=1)

    # Character
    grid_char = get_grid_character(
        game_state.entity_manager.character,
        layout.COMBAT_ACTOR_HEIGHT,
        layout.COMBAT_ACTOR_WIDTH,
        layout.COMBAT_ACTOR_MARGIN_BAR,
        layout.COMBAT_ACTOR_MARGIN_NAME,
    )
    y_char = y_hand - len(grid_char) - layout.COMBAT_CARD_HAND_PUSH + 1
    grid_main = paste_grid(grid_char, grid_main, y_char, x=1)

    # Monsters
    grid_monsters, height_monsters = _get_grid_monsters(
        game_state.entity_manager.monsters,
        game_state.entity_manager.character,
        layout.COMBAT_ACTOR_HEIGHT,
        layout.COMBAT_ACTOR_WIDTH,
        2 * layout.COMBAT_TARGET_MARGIN_X + 1,
        layout.COMBAT_ACTOR_MARGIN_BAR,
        layout.COMBAT_ACTOR_MARGIN_NAME,
    )
    grid_main = paste_grid(grid_monsters, grid_main, y=1, x=width - len(grid_monsters[0]))

    # Card target
    if game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD:
        num_monsters = len(game_state.entity_manager.monsters)
        height = height_monsters[idx_hover]
        grid_border_target = _get_grid_border_target(
            height + 2 * layout.COMBAT_TARGET_MARGIN_Y,
            layout.COMBAT_ACTOR_WIDTH + 2 * layout.COMBAT_TARGET_MARGIN_X,
            margin_y=2,
            margin_x=4,
            color_code=color.FG_GREEN_BOLD,
        )
        grid_main = paste_grid(
            grid_border_target,
            grid_main,
            y=0,
            x=width
            - (num_monsters - idx_hover)
            * (layout.COMBAT_ACTOR_WIDTH + 2 * layout.COMBAT_TARGET_MARGIN_X + 1)
            - layout.COMBAT_TARGET_MARGIN_X,  # TODO: maybe improve legibility
            ignore_bg=True,
        )

    # Energy
    grid_energy = _get_grid_energy(game_state.entity_manager.energy)
    grid_main = paste_grid(grid_energy, grid_main, y=1, x=1)

    return grid_main


def _get_grid_hand(
    hand: list[EntityCard],
    height_card: int,
    width_card: int,
    gap_x: int,
    margin_push: int,
    idx_push: int,
) -> Grid:
    num_card = len(hand)

    # Initalize grid
    grid = init_grid(height_card + margin_push, num_card * (width_card + gap_x))

    for idx, card in enumerate(hand):
        if idx == idx_push:
            y = 0
            color_code = color.FG_GREEN_BOLD
        else:
            y = margin_push
            color_code = None

        grid_card = get_grid_card(card, height_card, width_card, color_code)
        grid = paste_grid(grid_card, grid, y, idx * (width_card + gap_x))

    return grid


def _get_grid_border_target(
    height: int, width: int, margin_y: int, margin_x: int, color_code: str | None = None
) -> Grid:
    # Initialize grid and put the border
    grid = init_grid(height, width)
    grid = put_border(grid, color_code)

    # Erase the middle sections
    len_erase_h = width - 2 * margin_x
    grid = put_str(grid, " " * len_erase_h, y=0, x=margin_x)
    grid = put_str(grid, " " * len_erase_h, y=height - 1, x=margin_x)

    # Erase vertical TODO: maybe improve
    for y, row in enumerate(grid):
        for x in range(len(row)):
            if x == 0 and y >= margin_y and y <= height - 1 - margin_y:
                row[x] = " "
            elif x == width - 1 and y >= margin_y and y <= height - 1 - margin_y:
                row[x] = " "

    return grid


def _get_grid_monsters(
    monsters: list[EntityMonster],
    character: EntityCharacter,
    height_monster: int,
    width_monster: int,
    x_gap: int,
    margin_bar: int,
    margin_name: int,
) -> tuple[Grid, list[int]]:
    num_monster = len(monsters)

    # Initialize grid
    heights = [get_grid_actor_height(monster, height_monster) for monster in monsters]
    height_max = max(heights)
    grid = init_grid(height_max, num_monster * (width_monster + x_gap))

    for idx, monster in enumerate(monsters):
        grid_monster = get_grid_monster(
            monster, character, height_monster, width_monster, margin_bar, margin_name
        )
        grid = paste_grid(grid_monster, grid, y=0, x=idx * (width_monster + x_gap))

    return grid, heights


def _get_grid_energy(energy: EntityEnergy) -> Grid:
    grid = init_grid(5, 5)

    # Fill
    grid = put_str(grid, " ", y=0, x=1, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=0, x=2, color_code=color.BG_SILVER)

    grid = put_str(grid, " ", y=1, x=0, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=1, x=1, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=1, x=2, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=1, x=3, color_code=color.BG_SILVER)

    grid = put_str(grid, " ", y=2, x=0, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=2, x=4, color_code=color.BG_SILVER)

    grid = put_str(grid, " ", y=3, x=1, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=3, x=4, color_code=color.BG_SILVER)

    grid = put_str(grid, " ", y=4, x=2, color_code=color.BG_SILVER)
    grid = put_str(grid, " ", y=4, x=3, color_code=color.BG_SILVER)

    grid = put_str(
        grid,
        f"{energy.current}",
        y=1,
        x=1,
        color_code=color.BG_DARK_GREEN if energy.current >= 3 else None,
    )
    grid = put_str(
        grid, " ", y=1, x=2, color_code=color.BG_DARK_GREEN if energy.current >= 3 else None
    )
    grid = put_str(
        grid, " ", y=2, x=1, color_code=color.BG_DARK_GREEN if energy.current >= 2 else None
    )
    grid = put_str(
        grid, "/", y=2, x=2, color_code=color.BG_DARK_GREEN if energy.current >= 2 else None
    )
    grid = put_str(
        grid, " ", y=2, x=3, color_code=color.BG_DARK_GREEN if energy.current >= 2 else None
    )
    grid = put_str(
        grid, " ", y=3, x=2, color_code=color.BG_DARK_GREEN if energy.current >= 1 else None
    )
    grid = put_str(
        grid, " ", y=3, x=3, color_code=color.BG_DARK_GREEN if energy.current >= 1 else None
    )
    grid = put_str(
        grid,
        f"{energy.max}",
        y=3,
        x=3,
        color_code=color.BG_DARK_GREEN if energy.current >= 1 else None,
    )

    return grid
