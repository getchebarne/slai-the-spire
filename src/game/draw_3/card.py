from src.game.draw_3.grid import Grid
from src.game.draw_3.color import FG_GREEN_BOLD
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import paste_grid
from src.game.draw_3.grid import put_border
from src.game.draw_3.grid import put_hline
from src.game.draw_3.grid import put_str
from src.game.entity.card import EntityCard


def get_grid_card(
    entity_card: EntityCard, height: int, width: int, color_code: str | None = None
) -> Grid:
    # Initialize grid
    grid = init_grid(height, width)

    # Border
    grid = put_border(grid, color_code)

    # Name
    grid = _put_card_name(grid, entity_card.name, 3, color_code)

    # Text box
    grid = _put_description_box(grid, None, 10, color_code)

    return grid


def get_grid_card_zoomed(
    entity_card: EntityCard, height: int, width: int, color_code: str | None = None
) -> Grid:
    # Initialize grid
    grid = init_grid(height, width)

    # Border
    grid = put_border(grid, color_code)

    # Name
    grid = _put_card_name(grid, entity_card.name, 3, color_code)

    # Text box
    desc = "Deal 6 damage."
    grid = _put_description_box(grid, desc, 10, color_code)

    return grid


# def get_grid_card_paginate(
#     entity_cards: list[EntityCard],
#     num_rows: int,
#     num_cols: int,
#     height_card: int,
#     width_card: int,
#     y_gap: int,
#     offset_y: int,
#     offset_x: int,
# ) -> Grid:
#     # Initalize grid
#     x_gap = 2 * y_gap
#     width_scroll = 2
#     grid = init_grid(
#         num_rows * height_card + (num_rows + 1) * y_gap,
#         num_cols * width_card + (num_cols + 1) * x_gap + width_scroll,
#     )

#     # Put border
#     grid = put_border(grid)

#     rows_max = (len(entity_cards) - 1) // num_cols + 1
#     scroll_levels = (rows_max - 1) // num_rows

#     # Fix offset TODO move
#     offset_y = max(0, offset_y)
#     offset_y = min(scroll_levels, offset_y)

#     # Scroll bar
#     grid_scroll = init_grid(len(grid) - 2, width=3)
#     ratio = num_rows / rows_max
#     scroll_height = len(grid_scroll)
#     scroll_thumb_height = max(1, int(scroll_height * ratio))  # Ensure at least 1 block
#     scroll_position = int(offset_y / scroll_levels * (scroll_height - scroll_thumb_height))

#     for y, row in enumerate(grid_scroll):
#         for x in range(len(row)):
#             if scroll_position <= y < scroll_position + scroll_thumb_height and x == 1:
#                 row[x] = "█"

#     grid_scroll = put_border(grid_scroll)

#     grid = paste_grid(grid_scroll, grid, y=1, x=len(grid[0]) - 4)

#     # Iterate
#     for idx, card in enumerate(entity_cards):

#         # Calculate row
#         row = idx // num_cols
#         if row < offset_y or row >= offset_y + num_rows:
#             continue

#         # Calculate column
#         col = idx % num_cols

#         # Normalize row
#         row -= offset_y

#         grid_card = get_grid_card(card, height_card, width_card)

#         # Paste
#         grid = paste_grid(
#             grid_card,
#             grid,
#             y=row * (height_card + y_gap) + y_gap,
#             x=col * (width_card + x_gap) + x_gap,
#         )

#     return grid


def get_grid_card_paginate(
    entity_cards: list[EntityCard],
    num_rows: int,
    num_cols: int,
    height_card: int,
    width_card: int,
    y_gap: int,
    offset_y: int,
    offset_x: int,
) -> Grid:
    # Initalize grid
    x_gap = 2 * y_gap
    grid = init_grid(
        num_rows * height_card + (num_rows + 1) * y_gap,
        num_cols * width_card + (num_cols + 1) * x_gap,
    )

    # Put border
    grid = put_border(grid)

    # Fix offset TODO move
    max_rows = 1 + (len(entity_cards) - 1) // num_cols
    offset_y = max(0, offset_y)
    offset_y = min(max_rows - 1, offset_y)

    # Iterate
    for idx, card in enumerate(entity_cards):
        row = idx // num_cols
        if row > offset_y - num_rows and row < max(num_rows, offset_y + 1):
            # Calculate column
            col = idx % num_cols

            row_src = row
            if offset_y >= num_rows:
                row = row + num_rows - offset_y - 1

            if col == offset_x and row_src == offset_y:
                color_code = FG_GREEN_BOLD
            else:
                color_code = None

            grid_card = get_grid_card(card, height_card, width_card, color_code)

            # Paste
            grid = paste_grid(
                grid_card,
                grid,
                y=row * (height_card + y_gap) + y_gap,
                x=col * (width_card + x_gap) + x_gap,
            )

    return grid


def _put_card_name(grid: Grid, name: str, y_factor: int, color_code: str | None = None) -> Grid:
    height = len(grid)
    width = len(grid[0])

    name_len = len(name)
    y_name = height // y_factor
    x_name = (width - name_len) // 2
    grid = put_str(grid, name, y_name, x_name, color_code)
    grid = put_hline(grid, y_name + 1, x_name, name_len, color_code)

    return grid


def _put_description_box(
    grid: Grid, description: str | None, y_factor: int, color_code: str | None = None
) -> Grid:
    height = len(grid)
    width = len(grid[0])

    margin_y = height // y_factor
    margin_x = 2 * margin_y
    text_height = height // 2 - margin_y
    text_width = width - 2 * margin_x

    grid_text = init_grid(text_height, text_width)
    grid_text = put_border(grid_text, color_code)
    if description is not None:
        description_lines = description.split("\n")
        y_start = (text_height - len(description_lines)) // 2
        for idx, description_line in enumerate(description_lines):
            grid_text = put_str(
                grid_text,
                description_line,
                y_start + idx,
                (text_width - len(description_line)) // 2,
                color_code,
            )

    grid = paste_grid(grid_text, grid, height // 2, margin_x)

    return grid
