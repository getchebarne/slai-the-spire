import numpy as np
import torch

from src.game.const import MAP_HEIGHT
from src.game.const import MAP_WIDTH
from src.game.entity.map_node import RoomType  # TODO: should only import from `view`
from src.game.view.map_ import ViewMap


_ROOM_TYPE_CHANNEL = {room_type: channel for channel, room_type in enumerate(RoomType)}
_ROOM_TYPE_NUM = len(RoomType)
_NUM_CHANNELS = _ROOM_TYPE_NUM + MAP_WIDTH + 1


def get_encoding_map_dim() -> tuple[int, int, int]:
    return (MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS)


def _encode_view_map_into(out: np.ndarray, view_map: ViewMap) -> None:
    """Encode a map directly into a pre-allocated numpy array.

    out shape: (MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS)
    """
    # Populate room type and edge channels
    for y, row in enumerate(view_map.nodes):
        for x, node in enumerate(row):
            if node is None:
                continue

            # One-hot encode the room type
            idx_room_type = _ROOM_TYPE_CHANNEL[node.room_type]
            out[y, x, idx_room_type] = 1.0

            # Multi-hot encode the outgoing edges/paths
            if node.x_next is not None:
                for x_next in node.x_next:
                    if 0 <= x_next < MAP_WIDTH:
                        idx_edge = _ROOM_TYPE_NUM + x_next
                        out[y, x, idx_edge] = 1.0

    # Populate the current position channel
    if (
        view_map.y_current is not None
        and view_map.x_current is not None
        and 0 <= view_map.y_current < MAP_HEIGHT
        and 0 <= view_map.x_current < MAP_WIDTH
    ):
        idx_current_pos = _NUM_CHANNELS - 1
        out[view_map.y_current, view_map.x_current, idx_current_pos] = 1.0


def encode_batch_view_map(batch_view_map: list[ViewMap], device: torch.device) -> torch.Tensor:
    """Encode a batch of maps using NumPy pre-allocation."""
    batch_size = len(batch_view_map)

    # Pre-allocate numpy array
    x_out = np.zeros((batch_size, MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS), dtype=np.float32)

    for b, view_map in enumerate(batch_view_map):
        _encode_view_map_into(x_out[b], view_map)

    return torch.from_numpy(x_out).to(device)
