import random
from dataclasses import dataclass

from src.game.types_ import RoomType


_MAP_WIDTH = 7
_MAP_HEIGHT = 15
_PATH_DENSITY = 6
_ANCESTOR_GAP_MIN = 3
_ANCESTOR_GAP_MAX = 5


@dataclass
class MapNode:
    x: int
    y: int
    room_type: RoomType | None = None


@dataclass(frozen=True)
class MapEdge:
    x_source: int
    y_source: int
    x_target: int
    y_target: int


def _generate_map(
    map_width: int = _MAP_WIDTH, map_height: int = _MAP_HEIGHT, path_density: int = _PATH_DENSITY
) -> list[list[None]]:
    x_start_first = None
    map_edges = []
    for d in range(path_density):
        x_start = random.randint(0, map_width - 1)
        if d == 0:
            x_start_first = x_start

        while x_start == x_start_first and d == 1:
            x_start = random.randint(0, map_width - 1)

        _generate_edge(MapNode(x_start, 0), map_edges, map_width, map_height)

    map_edges = _trim_redundant_edges_first_to_second_row(map_edges)
    print_map_graph(map_edges, map_width, map_height)


# TODO: add boss?
def _generate_edge(
    node_current: MapNode, map_edges: list[MapEdge], map_width: int, map_height: int
) -> None:
    if node_current.y == map_height - 1:
        return

    if node_current.x == 0:
        offset_x = random.randint(0, 1)

    elif node_current.x == map_width - 1:
        offset_x = random.randint(-1, 0)

    else:
        offset_x = random.randint(-1, 1)

    # Create target node candidate
    node_target = MapNode(node_current.x + offset_x, node_current.y + 1)

    # Get target node's parents
    node_parents_target = _get_node_parents(node_target, map_edges)

    # Iterate
    for node_parent in node_parents_target:
        if not _do_nodes_have_the_same_coordinates(node_current, node_parent):
            # Get common ancestors
            node_ancestor = _get_common_ancestor(
                node_parent, node_current, map_edges, _ANCESTOR_GAP_MAX
            )
            if node_ancestor is not None:
                ancestor_gap = node_target.y - node_ancestor.y
                if ancestor_gap < _ANCESTOR_GAP_MIN:
                    if node_target.x > node_current.x:
                        node_target.x = node_current.x + random.randint(-1, 0)

                    elif node_target.x == node_current.x:
                        node_target.x = node_current.x + random.randint(-1, 1)

                    else:
                        node_target.x = node_current.x + random.randint(0, 1)

                    node_target.x = min(max(node_target.x, 0), map_width - 1)

    # Trimming to prevent path overlap
    if node_current.x > 0:
        for map_edge in map_edges:
            if map_edge.y_source == node_current.y and map_edge.x_source == node_current.x - 1:
                # Left node
                if map_edge.x_target > node_target.x:
                    node_target.x = map_edge.x_target

    if node_current.x < map_width - 1:
        for map_edge in map_edges:
            if map_edge.y_source == node_current.y and map_edge.x_source == node_current.x + 1:
                # Right node
                if map_edge.x_target < node_target.x:
                    node_target.x = map_edge.x_target

    # Append new edge
    map_edges.append(MapEdge(node_current.x, node_current.y, node_target.x, node_target.y))

    return _generate_edge(node_target, map_edges, map_width, map_height)


# TODO: make more efficient w/ sorting
def _get_node_parents(node: MapNode, map_edges: list[MapEdge]) -> list[MapNode]:
    parents = []
    for map_edge in map_edges:
        if map_edge.x_target == node.x and map_edge.y_target == node.y:
            parents.append(MapNode(map_edge.x_source, map_edge.y_source))

    return parents


def _get_common_ancestor(
    node_1: MapNode, node_2: MapNode, map_edges: list[MapEdge], ancestor_gap_max: int
) -> MapNode | None:
    if node_1 == node_2:
        raise ValueError("Can't get common ancestor two identical nodes")

    if node_1.y != node_2.y:
        raise ValueError("Can't get common ancestor for nodes that aren't on the same y-level")

    # Note: this should compare the y-coordinates of both nodes, but this is how it's implemented
    # in the official game's code. It seems to work anyway
    if node_1.x < node_2.y:
        node_left = node_1
        node_right = node_2

    else:
        node_left = node_2
        node_right = node_1

    node_parents_left = _get_node_parents(node_left, map_edges)
    if not node_parents_left:
        return None

    node_parents_right = _get_node_parents(node_right, map_edges)
    if not node_parents_right:
        return None

    y_current = node_1.y
    while (y_current >= 0) and (y_current >= node_1.y - ancestor_gap_max):
        node_left = _get_node_with_max_x(node_parents_left)
        node_right = _get_node_with_min_x(node_parents_right)

        if node_left == node_right:
            return node_left

        y_current -= 1


def _trim_redundant_edges_first_to_second_row(map_edges: list[MapEdge]):
    seen_targets = set()
    map_edges_trimmed = []

    for map_edge in map_edges:
        if map_edge.y_source == 0 and map_edge.y_target == 1:
            if map_edge.x_target in seen_targets:
                continue  # Redundant map_edge, skip it

            seen_targets.add(map_edge.x_target)

        map_edges_trimmed.append(map_edge)

    return map_edges_trimmed


def print_map_graph(edges: list[MapEdge], map_width: int, map_height: int):
    grid_rows = map_height * 2
    grid_cols = map_width * 3
    grid = [[" " for _ in range(grid_cols)] for _ in range(grid_rows)]

    # Determine active nodes (used as source or target in any edge)
    active_nodes = set()
    for edge in edges:
        active_nodes.add((edge.x_source, edge.y_source))
        active_nodes.add((edge.x_target, edge.y_target))

    # Place only active nodes
    for y in range(map_height):
        for x in range(map_width):
            if (x, y) in active_nodes:
                gx = x * 3 + 1  # center column
                gy = y * 2  # top row of the 2-row cell
                grid[gy][gx] = "O"

    # Place all edges
    for edge in edges:
        dx = edge.x_target - edge.x_source
        dy = edge.y_target - edge.y_source
        if dx not in (-1, 0, 1) or dy != 1:
            continue  # skip invalid edge

        gx = edge.x_source * 3 + 1
        gy = edge.y_source * 2
        edge_row = gy + 1

        if dx == -1:
            grid[edge_row][gx - 1] = "\\"
        elif dx == 0:
            grid[edge_row][gx] = "|"
        elif dx == 1:
            grid[edge_row][gx + 1] = "/"

    # Print the grid
    for row in reversed(grid):
        print("".join(row))


def _get_node_with_max_x(nodes: list[MapNode]) -> MapNode:
    return max(nodes, key=lambda node: node.x)


def _get_node_with_min_x(nodes: list[MapNode]) -> MapNode:
    return min(nodes, key=lambda node: node.x)


def _do_nodes_have_the_same_coordinates(node_1: MapNode, node_2: MapNode) -> bool:
    return node_1.x == node_2.x and node_1.y == node_2.y


if __name__ == "__main__":
    _generate_map()
    pass
