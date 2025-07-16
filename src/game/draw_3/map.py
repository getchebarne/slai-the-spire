from src.game.draw_3.color import FG_GREEN_BOLD
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import put_str
from src.game.entity.map_node import EntityMapNode
from src.game.entity.map_node import RoomType


_ROOM_TYPE_CHAR = {
    RoomType.COMBAT_MONSTER: "M",
    RoomType.REST_SITE: "R",
}
_X_DELTA_CHAR = {
    -1: "\\",
    0: "|",
    1: "/",
}


def get_grid_map(
    map_nodes: list[list[EntityMapNode | None]],
    map_node_active: EntityMapNode | None,
    idx_hover: int,
) -> Grid:
    # Get grid dimensions
    map_height = len(map_nodes)
    map_width = len(map_nodes[0])
    map_grid_height = 2 * map_height
    map_grid_width = 3 * map_width

    # Get next floor coordinate
    y_next = 0 if map_node_active is None else map_node_active.y + 1
    if map_node_active is None:
        x_next = [x for x, node in enumerate(map_nodes[0]) if node is not None]
    else:
        x_next = sorted(list(map_node_active.x_next))

    # Intialize grid
    grid = init_grid(map_grid_height, map_grid_width)

    # Iterate over floors and nodes
    for y, floor in enumerate(reversed(map_nodes)):
        for x, node in enumerate(floor):
            if node is None:
                continue

            # Node
            color_code = None
            if map_height - 1 - y == y_next and x == x_next[idx_hover]:
                color_code = FG_GREEN_BOLD

            grid = put_str(grid, _ROOM_TYPE_CHAR[node.room_type], 2 * y + 1, 3 * x + 1, color_code)

            # Edges
            for x_edge in node.x_next:
                x_delta = x_edge - x
                ch = _X_DELTA_CHAR[x_delta]
                grid = put_str(grid, ch, 2 * y, 3 * x + 1 + x_delta)

    return grid
