import torch

from src.game.const import MAP_HEIGHT
from src.game.const import MAP_WIDTH
from src.game.entity.map_node import RoomType  # TODO: should only import from `view`
from src.game.view.map_ import ViewMap


_ROOM_TYPE_CHANNEL = {room_type: channel for channel, room_type in enumerate(RoomType)}
_ROOM_TYPE_NUM = len(RoomType)


def _get_view_map_dummy() -> ViewMap:
    return ViewMap([[None] * MAP_WIDTH] * MAP_HEIGHT)


def get_encoding_map_dim() -> tuple[int, int, int]:
    view_map_dummy = _get_view_map_dummy()
    encoding_map_dummy = encode_view_map(view_map_dummy, torch.device("cpu"))
    return tuple(encoding_map_dummy.shape)


def encode_view_map(view_map: ViewMap, device: torch.device) -> torch.Tensor:
    map_height = len(view_map.nodes)
    map_width = len(view_map.nodes[0])

    # Calculate total channels and initialize the encoding tensor
    num_channels = _ROOM_TYPE_NUM + map_width + 1
    encoding = torch.zeros(
        (map_height, map_width, num_channels), dtype=torch.float32, device=device
    )

    # Populate each channel
    for y, row in enumerate(view_map.nodes):
        for x, node in enumerate(row):
            if node is None:
                # The feature vector remains all zeros
                continue

            # RoomType
            idx_room_type = _ROOM_TYPE_CHANNEL[node.room_type]
            encoding[y, x, idx_room_type] = 1.0

            # Edges
            if node.x_next is not None:
                for x_next in node.x_next:
                    idx_edge = _ROOM_TYPE_NUM + x_next
                    encoding[y, x, idx_edge] = 1.0

    # Current position
    if (
        view_map.y_current is not None
        and view_map.x_current is not None
        and view_map.y_current < MAP_HEIGHT  # TODO: improve this garbage
    ):
        idx_current_pos = num_channels - 1
        encoding[view_map.y_current, view_map.x_current, idx_current_pos] = 1.0

    return encoding
