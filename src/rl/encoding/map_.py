import torch

from src.game.const import MAP_HEIGHT
from src.game.const import MAP_WIDTH
from src.game.entity.map_node import RoomType  # TODO: should only import from `view`
from src.game.view.map_ import ViewMap


_ROOM_TYPE_CHANNEL = {room_type: channel for channel, room_type in enumerate(RoomType)}
_ROOM_TYPE_NUM = len(RoomType)
_NUM_CHANNELS = _ROOM_TYPE_NUM + MAP_WIDTH + 1


def _get_view_map_dummy() -> ViewMap:
    return ViewMap([[None] * MAP_WIDTH] * MAP_HEIGHT)


def get_encoding_map_dim() -> tuple[int, int, int]:
    return (MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS)


def _encode_view_map(view_map: ViewMap) -> list[list[list[float]]]:
    # Initialize a 3D list of zeros: height x width x channels
    encoding = [
        [[0.0 for _ in range(_NUM_CHANNELS)] for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)
    ]

    # Populate room type and edge channels
    for y, row in enumerate(view_map.nodes):
        for x, node in enumerate(row):
            if node is None:
                continue

            # One-hot encode the room type
            idx_room_type = _ROOM_TYPE_CHANNEL[node.room_type]
            encoding[y][x][idx_room_type] = 1.0

            # Multi-hot encode the outgoing edges/paths
            if node.x_next is not None:
                for x_next in node.x_next:
                    idx_edge = _ROOM_TYPE_NUM + x_next
                    encoding[y][x][idx_edge] = 1.0

    # Populate the current position channel
    if (
        view_map.y_current is not None
        and view_map.x_current is not None
        and view_map.y_current < MAP_HEIGHT
    ):
        y_curr, x_curr = view_map.y_current, view_map.x_current
        idx_current_pos = _NUM_CHANNELS - 1
        encoding[y_curr][x_curr][idx_current_pos] = 1.0

    return encoding


def encode_batch_view_map(batch_view_map: list[ViewMap], device: torch.device) -> torch.Tensor:
    # Generate a list of encodings, one for each map in the batch
    batch_encodings_list = [_encode_view_map(view_map) for view_map in batch_view_map]

    # Convert the list of 3D lists into a single 4D PyTorch tensor
    return torch.tensor(batch_encodings_list, dtype=torch.float32, device=device)
