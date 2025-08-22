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
    view_map = ViewMap(deepcopy(entity_manager.id_map_nodes))

    for y, row in enumerate(view_map.nodes):
        for x, id_node in enumerate(row):
            if id_node is not None:
                map_node = entity_manager.entities[id_node]
                view_map.nodes[y][x] = ViewMapNode(map_node.room_type, map_node.x_next)

    if entity_manager.id_map_node_active is None:
        return view_map

    # Get current node coordinates
    map_node_active = entity_manager.entities[entity_manager.id_map_node_active]
    view_map = replace(view_map, y_current=map_node_active.y, x_current=map_node_active.x)

    return view_map
