import random

from src.game.entity.map_node import EntityMapNode
from src.game.entity.map_node import RoomType


_MAP_HEIGHT = 15
_MAP_WIDTH = 7
_PATH_DENSITY = 6
_ANCESTOR_GAP_MIN = 3
_ANCESTOR_GAP_MAX = 5
_FACTOR_NUM_REST_SITE = 0.25


def generate_map(
    map_height: int = _MAP_HEIGHT, map_width: int = _MAP_WIDTH, path_density: int = _PATH_DENSITY
) -> list[list[EntityMapNode | None]]:
    # Initialize empty map
    nodes = _initialize_nodes(map_height, map_width)

    x_source_first = None
    for d in range(path_density):
        x_source = random.randint(0, map_width - 1)
        if d == 0:
            x_source_first = x_source

        while x_source == x_source_first and d == 1:
            x_source = random.randint(0, map_width - 1)

        y_source = 0
        if nodes[y_source][x_source] is None:
            nodes[y_source][x_source] = EntityMapNode(y_source, x_source)

        node_source = nodes[y_source][x_source]
        while node_source.y < map_height - 1:
            # Create the node
            node_target = _create_node(node_source, nodes)

            # If the node hasn't been added to the map yet, do so
            if nodes[node_target.y][node_target.x] is None:
                nodes[node_target.y][node_target.x] = node_target

            # Add it to the source node's connected nodes
            nodes[node_source.y][node_source.x].x_next.add(node_target.x)

            # Overwrite the source node for the next iteration
            node_source = node_target

    # Trim redunant edges sourcing from the first row
    nodes = _trim_redundant_edges_first_to_second_floor(nodes)

    # Assign room types
    _assign_room_types(nodes)

    return nodes


def _initialize_nodes(map_height: int, map_width: int) -> list[list[EntityMapNode | None]]:
    nodes = []
    for _ in range(map_height):
        nodes.append([None] * map_width)

    return nodes


def _create_node(
    node_source: EntityMapNode, nodes: list[list[EntityMapNode | None]]
) -> EntityMapNode | None:
    map_height = len(nodes)
    map_width = len(nodes[0])
    y_source = node_source.y
    x_source = node_source.x

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
    node_target = EntityMapNode(y_target, x_target)

    # Get target node's parents and iterate over them
    target_parents = _get_node_parents(node_target, nodes)
    for node_target_parent in target_parents:
        if node_target_parent.y == y_source and node_target_parent.x == x_source:
            continue

        # Get common ancestors
        node_ancestor = _get_common_ancestor(
            node_target_parent, node_source, nodes, _ANCESTOR_GAP_MAX
        )
        if node_ancestor is not None:
            ancestor_gap = y_target - node_ancestor.y
            if ancestor_gap < _ANCESTOR_GAP_MIN:
                if x_target > x_source:
                    x_target = x_source + random.randint(-1, 0)

                elif x_target == x_source:
                    x_target = x_source + random.randint(-1, 1)

                else:
                    x_target = x_source + random.randint(0, 1)

                x_target = min(max(x_target, 0), map_width - 1)

    # Trim to prevent path overlap - from left to right)
    if x_source > 0:
        x_source_left = x_source - 1
        node_left = nodes[y_source][x_source_left]
        if node_left is not None:
            for x_target_left in node_left.x_next:
                if x_target_left > x_target:
                    x_target = x_target_left

    # Right to left
    if x_source < map_width - 1:
        x_source_right = x_source + 1
        node_right = nodes[y_source][x_source_right]
        if node_right is not None:
            for x_target_right in node_right.x_next:
                if x_target_right < x_target:
                    x_target = x_target_right

    # Return new node
    return EntityMapNode(y_target, x_target)


def _get_node_parents(
    node_query: EntityMapNode, nodes: list[list[EntityMapNode | None]]
) -> list[EntityMapNode]:
    if node_query.y == 0:
        return []

    parents = []
    y_parent = node_query.y - 1
    for node_parent_candidate in nodes[y_parent]:
        if node_parent_candidate is None:
            continue

        if node_query.x in node_parent_candidate.x_next:
            parents.append(node_parent_candidate)

    return parents


def _get_common_ancestor(
    node_1: EntityMapNode,
    node_2: EntityMapNode,
    nodes: list[list[EntityMapNode | None]],
    ancestor_gap_max: int,
) -> EntityMapNode | None:
    if node_1.y != node_2.y:
        raise ValueError("Can't get common ancestor for nodes that aren't on the same y-level")

    if node_1.x == node_2.x:
        raise ValueError("Can't get common ancestor two identical nodes")

    # Note: this should compare the y-coordinates of both nodes, but this is how it's implemented
    # in the official game's code. It seems to work anyway
    if node_1.x < node_2.y:
        node_l = node_1
        node_r = node_2

    else:
        node_l = node_2
        node_r = node_1

    node_parents_left = _get_node_parents(node_l, nodes)
    if not node_parents_left:
        return None

    node_parents_right = _get_node_parents(node_r, nodes)
    if not node_parents_right:
        return None

    y_current = node_1.y
    while (y_current >= 0) and (y_current >= node_1.y - ancestor_gap_max):
        node_left = max(node_parents_left, key=lambda node: node.x)
        node_right = min(node_parents_right, key=lambda node: node.x)

        if node_left == node_right:
            return node_left

        y_current -= 1


def _trim_redundant_edges_first_to_second_floor(
    nodes: list[list[EntityMapNode | None]],
) -> list[list[EntityMapNode | None]]:
    x_seen = set()
    x_remove = set()
    for x_source, node in enumerate(nodes[0]):
        if node is None:
            continue

        x_next_filtered = set()
        for x_next in node.x_next:
            if x_next in x_seen:
                continue

            x_next_filtered.add(x_next)
            x_seen.add(x_next)

        if not x_next_filtered:
            x_remove.add(x_source)

        node.x_next = x_next_filtered

    # Remove nodes without targets
    for x_r in x_remove:
        nodes[0][x_r] = None

    return nodes


# TODO: this is placeholder logic
def _assign_room_types(nodes: list[list[EntityMapNode | None]]) -> None:
    nodes_flat = [node for row in nodes for node in row if node is not None]
    num_nodes = len(nodes_flat)
    room_types = [RoomType.COMBAT_MONSTER] * num_nodes
    room_types[: int(_FACTOR_NUM_REST_SITE * num_nodes)] = [RoomType.REST_SITE] * int(
        _FACTOR_NUM_REST_SITE * num_nodes
    )
    random.shuffle(room_types)

    for node, room_type in zip(nodes_flat, room_types):
        node.room_type = room_type

    # Make sure last floor is all rest sites
    for map_node in nodes[-1]:
        if map_node is None:
            continue

        map_node.room_type = RoomType.REST_SITE
