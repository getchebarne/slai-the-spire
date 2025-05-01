import random
from typing import TypeAlias

from src.game.entity.map_node import EntityMapNode
from src.game.entity.map_node import RoomType


Map: TypeAlias = dict[int, dict[int, EntityMapNode]]

_MAP_HEIGHT = 15
_MAP_WIDTH = 7
_PATH_DENSITY = 6
_ANCESTOR_GAP_MIN = 3
_ANCESTOR_GAP_MAX = 5
_FACTOR_NUM_REST_SITE = 0.25


def generate_map(
    map_height: int = _MAP_HEIGHT, map_width: int = _MAP_WIDTH, path_density: int = _PATH_DENSITY
) -> Map:
    # Initialize empty map
    map_ = _initialize_map(map_height)

    x_source_first = None
    for d in range(path_density):
        x_source = random.randint(0, map_width - 1)
        if d == 0:
            x_source_first = x_source

        while x_source == x_source_first and d == 1:
            x_source = random.randint(0, map_width - 1)

        y_source = 0
        if x_source not in map_[y_source]:
            map_[y_source][x_source] = EntityMapNode(y_source, x_source)

        while y_source < map_height - 1:
            # Create the node
            y_target, x_target = _create_node(y_source, x_source, map_, map_height, map_width)

            # If the node hasn't been added to the map yet, do so
            if x_target not in map_[y_target]:
                map_[y_target][x_target] = EntityMapNode(y_target, x_target)

            # Add it to the source node's connected nodes
            map_[y_source][x_source].x_next.add(x_target)

            # Overwrite the source node for the next iteration
            y_source = y_target
            x_source = x_target

    # Trim redunant edges sourcing from the first row
    map_ = _trim_redundant_edges_first_to_second_floor(map_)

    # Assign room types
    _assign_room_types(map_)

    return map_


def _initialize_map(map_height: int) -> Map:
    return {y: {} for y in range(map_height)}


# TODO: add boss?
def _create_node(
    y_source: int, x_source: int, map_: Map, map_height: int, map_width: int
) -> tuple[int, int] | None:
    if y_source == map_height - 1:
        raise ValueError("Can't generate an edge for a node located at the map's upper limit")

    if x_source == 0:
        offset_x = random.randint(0, 1)

    elif x_source == map_width - 1:
        offset_x = random.randint(-1, 0)

    else:
        offset_x = random.randint(-1, 1)

    # Create target node candidate
    y_target = y_source + 1
    x_target = x_source + offset_x

    # Get target node's parents and iterate over them
    target_parents = _get_node_parents(y_target, x_target, map_)
    for y_target_parent, x_target_parent in target_parents:
        if y_target_parent == y_source and x_target_parent == x_source:
            continue

        # Get common ancestors
        node_ancestor = _get_common_ancestor(
            y_target_parent, x_target_parent, y_source, x_source, map_, _ANCESTOR_GAP_MAX
        )
        if node_ancestor is not None:
            y_ancestor, _ = node_ancestor
            ancestor_gap = y_target - y_ancestor
            if ancestor_gap < _ANCESTOR_GAP_MIN:
                if x_target > x_source:
                    x_target = x_source + random.randint(-1, 0)

                elif x_target == x_source:
                    x_target = x_source + random.randint(-1, 1)

                else:
                    x_target = x_source + random.randint(0, 1)

                x_target = min(max(x_target, 0), map_width - 1)

    # Trim to prevent path overlap - from left to right)
    x_map_node_data = map_[y_source]
    if x_source > 0:
        x_source_left = x_source - 1
        if x_source_left in x_map_node_data:
            for x_target_left in x_map_node_data[x_source_left].x_next:
                if x_target_left > x_target:
                    x_target = x_target_left

    # Right to left
    if x_source < map_width - 1:
        x_source_right = x_source + 1
        if x_source_right in x_map_node_data:
            for x_target_right in x_map_node_data[x_source_right].x_next:
                if x_target_right < x_target:
                    x_target = x_target_right

    # Return new edge
    return y_target, x_target


def _get_node_parents(y_query: int, x_query: int, map_: Map) -> list[tuple[int, int]]:
    if y_query == 0:
        return []

    parents = []
    y_parent = y_query - 1
    x_map_node_data = map_[y_parent]
    for x_other, map_node_data in x_map_node_data.items():
        if x_query in map_node_data.x_next:
            parents.append((y_parent, x_other))

    return parents


def _get_common_ancestor(
    y_1: int, x_1: int, y_2: int, x_2: int, map_: Map, ancestor_gap_max: int
) -> tuple[int, int] | None:
    if y_1 != y_2:
        raise ValueError("Can't get common ancestor for nodes that aren't on the same y-level")

    if x_1 == x_2:
        raise ValueError("Can't get common ancestor two identical nodes")

    # Note: this should compare the y-coordinates of both nodes, but this is how it's implemented
    # in the official game's code. It seems to work anyway
    if x_1 < y_2:
        y_l = y_1
        x_l = x_1
        y_r = y_2
        x_r = x_2

    else:
        y_l = y_2
        x_l = x_2
        y_r = y_1
        x_r = x_1

    node_parents_left = _get_node_parents(y_l, x_l, map_)
    if not node_parents_left:
        return None

    node_parents_right = _get_node_parents(y_r, x_r, map_)
    if not node_parents_right:
        return None

    y_current = y_1
    while (y_current >= 0) and (y_current >= y_1 - ancestor_gap_max):
        node_left = max(node_parents_left, key=lambda node: node[1])
        node_right = min(node_parents_right, key=lambda node: node[1])

        if node_left == node_right:
            return node_left

        y_current -= 1


def _trim_redundant_edges_first_to_second_floor(map_: Map) -> Map:
    x_seen = set()
    x_remove = set()
    for x_source, map_node_data in map_[0].items():
        x_next_filtered = set()
        for x_next in map_node_data.x_next:
            if x_next in x_seen:
                continue

            x_next_filtered.add(x_next)
            x_seen.add(x_next)

        if not x_next_filtered:
            x_remove.add(x_source)

        map_node_data.x_next = x_next_filtered

    # Remove nodes without targets
    for x_r in x_remove:
        del map_[0][x_r]

    return map_


def print_map(map_: Map, map_width: int, map_height: int) -> None:
    grid_rows = map_height * 2
    grid_cols = map_width * 3
    grid = [[" " for _ in range(grid_cols)] for _ in range(grid_rows)]

    # Place only active nodes
    for y in range(map_height):
        for x in range(map_width):
            try:
                # Place node
                map_node_data = map_[y][x]
                gx = x * 3 + 1  # center column
                gy = y * 2  # top row of the 2-row cell
                if map_node_data.room_type == RoomType.COMBAT_MONSTER:
                    grid[gy][gx] = "M"
                else:
                    grid[gy][gx] = "R"

                # Place edges
                gx = x * 3 + 1
                gy = y * 2
                edge_row = gy + 1
                for x_next in map_node_data.x_next:
                    dx = x_next - x

                    if dx == -1:
                        grid[edge_row][gx - 1] = "\\"
                    elif dx == 0:
                        grid[edge_row][gx] = "|"
                    elif dx == 1:
                        grid[edge_row][gx + 1] = "/"

            except KeyError:
                continue

    # Print the grid
    for row in reversed(grid):
        print("".join(row))


# TODO: this is placeholder logic
def _assign_room_types(map_: Map) -> None:
    map_node_data_flat = [
        map_node_data
        for _, x_map_node_data in map_.items()
        for _, map_node_data in x_map_node_data.items()
        if map_node_data is not None
    ]
    node_num = len(map_node_data_flat)
    room_types = [None] * node_num
    num_combat_monster = int((1 - _FACTOR_NUM_REST_SITE) * node_num)
    num_rest_site = node_num - num_combat_monster
    room_types[:num_combat_monster] = [RoomType.COMBAT_MONSTER] * num_combat_monster
    room_types[num_combat_monster:] = [RoomType.REST_SITE] * num_rest_site
    random.shuffle(room_types)

    for room_type, map_node_data in zip(room_types, map_node_data_flat):
        map_node_data.room_type = room_type
