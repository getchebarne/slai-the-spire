import copy
import re
from typing import TypeAlias


Grid: TypeAlias = list[list[str]]

# Regex to remove ANSI escape sequences
RE_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\033[0m"


def init_grid(height: int, width: int) -> Grid:
    return [[" "] * width for _ in range(height)]


def get_hw(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def put_border(grid: Grid, color_code: str | None = None) -> Grid:
    grid_copy = copy.deepcopy(grid)
    height, width = get_hw(grid)

    # Top border
    grid_copy[0][0] = _apply_color_code("┌", color_code)
    grid_copy[0][-1] = _apply_color_code("┐", color_code)
    for x in range(1, width - 1):
        grid_copy[0][x] = _apply_color_code("─", color_code)

    # Bottom border
    grid_copy[-1][0] = _apply_color_code("└", color_code)
    grid_copy[-1][-1] = _apply_color_code("┘", color_code)
    for x in range(1, width - 1):
        grid_copy[-1][x] = _apply_color_code("─", color_code)

    # Vertical borders
    for y in range(1, height - 1):
        grid_copy[y][0] = _apply_color_code("│", color_code)
        grid_copy[y][-1] = _apply_color_code("│", color_code)

    return grid_copy


def put_str(grid: Grid, str_: str, y: int, x: int, color_code: str | None = None) -> Grid:
    grid_copy = copy.deepcopy(grid)

    for x_offset, ch in enumerate(str_):
        if color_code is None:
            grid_copy[y][x + x_offset] = ch
        else:
            grid_copy[y][x + x_offset] = f"{color_code}{ch}{RESET}"

    return grid_copy


def put_hline(grid: Grid, y: int, x: int, len_: int, color_code: str | None = None) -> Grid:
    grid_copy = copy.deepcopy(grid)

    row = grid_copy[y]
    for x_offset in range(len_):
        row[x + x_offset] = _apply_color_code("─", color_code)

    return grid_copy


def paste_grid(grid_src: Grid, grid_dst: Grid, y: int, x: int, ignore_bg: bool = False) -> Grid:
    grid_dst_copy = copy.deepcopy(grid_dst)

    for y_offset, row in enumerate(grid_src):
        for x_offset, ch in enumerate(row):
            if ignore_bg and ch == " ":
                continue

            grid_dst_copy[y + y_offset][x + x_offset] = ch

    return grid_dst_copy


# TODO: put colors in colors.py
def set_color(grid: Grid, color_code: str = "\033[1;92m") -> Grid:
    grid_copy = copy.deepcopy(grid)

    for row in grid_copy:
        row[0] = f"{color_code}{_strip_ansi(row[0])}"
        row[-1] = f"{_strip_ansi(row[-1])}{RESET}"

    return grid_copy


def print_grid(grid: list[list[int]]) -> None:
    print(
        "\n".join(
            ["".join(row) for row in grid],
        )
    )


def _strip_ansi(text: str) -> str:
    return RE_ANSI_ESCAPE.sub("", text)


def _apply_color_code(ch: str, color_code: str) -> str:
    return f"{color_code}{ch}{RESET}" if color_code else ch
