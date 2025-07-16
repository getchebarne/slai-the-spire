from src.game.core.fsm import FSM
from src.game.draw_3.card import get_grid_card
from src.game.draw_3.color import FG_GREEN_BOLD
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import paste_grid
from src.game.draw_3.layout import GAP_CARD
from src.game.draw_3.layout import HEIGHT_CARD
from src.game.draw_3.layout import WIDTH_CARD
from src.game.state import GameState


def get_grid_combat_reward(game_state: GameState, idx_hover: int) -> Grid:
    num_cards = len(game_state.entity_manager.cards_reward)
    grid = init_grid(HEIGHT_CARD, WIDTH_CARD * num_cards + GAP_CARD * (num_cards - 1))

    for idx, card in enumerate(game_state.entity_manager.cards_reward):
        x = idx * (WIDTH_CARD + GAP_CARD)
        color_code = (
            None
            if idx != idx_hover or game_state.fsm == FSM.COMBAT_AWAIT_TARGET_CARD
            else FG_GREEN_BOLD
        )
        grid_card = get_grid_card(card, HEIGHT_CARD, WIDTH_CARD, color_code)
        grid = paste_grid(grid_card, grid, y=0, x=x)

    return grid
