import numpy as np
import slai
import torch

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH


# Snapshot RoomKind variants at module load. slai exposes 4 today
# (CombatMonster, CombatBoss, CombatElite, RestSite); slot count is fixed
# at module load. RoomKind is an enum.IntEnum (see slai's _to_intenum shim).
_ROOM_KIND_NAMES: list[str] = [m.name for m in slai.RoomKind]
_ROOM_KIND_TO_IDX: dict[object, int] = {m: idx for idx, m in enumerate(slai.RoomKind)}
_ROOM_KIND_NUM = len(_ROOM_KIND_NAMES)
_NUM_CHANNELS = _ROOM_KIND_NUM + MAP_WIDTH + 1  # room kind one-hot + edge multi-hot + cur-pos


def get_encoding_map_dim() -> tuple[int, int, int]:
    return (MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS)


def _encode_view_map_into(out: np.ndarray, view_map: slai.Map) -> None:
    """Encode a map directly into a pre-allocated numpy array.

    out shape: (MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS)
    """
    # Populate room kind and edge channels
    for y, row in enumerate(view_map.rooms[:MAP_HEIGHT]):
        for x, room in enumerate(row[:MAP_WIDTH]):
            if room is None:
                continue

            # One-hot encode the room kind
            idx_room_kind = _ROOM_KIND_TO_IDX.get(room.room_kind)
            if idx_room_kind is not None:
                out[y, x, idx_room_kind] = 1.0

            # Multi-hot encode the outgoing edges/paths
            for x_next in room.edges:
                if 0 <= x_next < MAP_WIDTH:
                    idx_edge = _ROOM_KIND_NUM + x_next
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


def encode_batch_view_map(batch_view_map: list[slai.Map], device: torch.device) -> torch.Tensor:
    """Encode a batch of maps using NumPy pre-allocation."""
    batch_size = len(batch_view_map)

    # Pre-allocate numpy array
    x_out = np.zeros((batch_size, MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS), dtype=np.float32)

    for b, view_map in enumerate(batch_view_map):
        _encode_view_map_into(x_out[b], view_map)

    return torch.from_numpy(x_out).to(device)
