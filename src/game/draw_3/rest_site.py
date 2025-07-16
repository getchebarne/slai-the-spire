from src.game.draw_3.color import FG_GREEN_BOLD
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import put_str
from src.game.draw_3.layout import WIDTH_MAIN
from src.game.state import GameState


def get_grid_rest_site(game_state: GameState, idx_hover: int, card_upgrade: bool) -> Grid:
    if card_upgrade:
        return _get_grid_rest_site_card_upgrade_true(game_state, idx_hover)

    return _get_grid_rest_site_card_upgrade_false(game_state, idx_hover)


def _get_grid_rest_site_card_upgrade_true(game_state: GameState, idx_hover: int) -> Grid:
    grid = init_grid(len(game_state.entity_manager.cards_in_deck), WIDTH_MAIN - 4)
    for idx, card in enumerate(game_state.entity_manager.cards_in_deck):
        color_code = None if idx != idx_hover else FG_GREEN_BOLD
        grid = put_str(grid, card.name, y=idx, x=0, color_code=color_code)

    return grid


def _get_grid_rest_site_card_upgrade_false(game_state: GameState, idx_hover: int) -> Grid:
    grid = init_grid(2, WIDTH_MAIN - 4)

    grid = put_str(grid, "Rest", y=0, x=0, color_code=None if idx_hover != 0 else FG_GREEN_BOLD)
    grid = put_str(grid, "Upgrade", y=1, x=0, color_code=None if idx_hover != 1 else FG_GREEN_BOLD)

    return grid
