from copy import deepcopy
from dataclasses import dataclass, replace

from src.game.entity.manager import EntityManager
from src.game.entity.map_node import RoomType


@dataclass(frozen=True)
class ViewMapNode:
    room_type: RoomType
    x_next: set[int] | None


@dataclass(frozen=True)
class ViewMap:
    nodes: list[list[ViewMapNode | None]]

    # Current node coordinates
    y_current: int | None = None
    x_current: int | None = None


def get_view_map(entity_manager: EntityManager) -> ViewMap:
    map_view = ViewMap(deepcopy(entity_manager.id_map_nodes))

    for y, row in enumerate(map_view.nodes):
        for x, id_node in enumerate(row):
            if id_node is not None:
                map_node = entity_manager.entities[id_node]
                map_view.nodes[y][x] = ViewMapNode(map_node.room_type, map_node.x_next)

    if entity_manager.id_map_node_active is None:
        return map_view

    # Get current node coordinates
    map_node_active = entity_manager.map_node_active
    map_view = replace(map_view, y_current=map_node_active.y, x_current=map_node_active.x)

    return map_view
