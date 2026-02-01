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
    # Build view nodes from entity map nodes
    view_nodes = []
    for row in entity_manager.map_nodes:
        view_row = []
        for node in row:
            if node is not None:
                view_row.append(ViewMapNode(node.room_type, node.x_next))
            else:
                view_row.append(None)
        view_nodes.append(view_row)

    view_map = ViewMap(view_nodes)

    if entity_manager.map_node_active is None:
        return view_map

    # Get current node coordinates
    map_node_active = entity_manager.map_node_active
    view_map = replace(view_map, y_current=map_node_active.y, x_current=map_node_active.x)

    return view_map
