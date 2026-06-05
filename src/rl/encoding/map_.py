import numpy as np
import torch
from slai import Map
from slai import RoomKind

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH


_ROOM_KIND_TO_IDX = {room_kind: i for i, room_kind in enumerate(RoomKind)}
_NUM_CHANNELS = (
    len(RoomKind)          # Room kind OHE
    + MAP_WIDTH            # Outgoing-edge multi-hot
    + 1                    # Current position
)

# Act-1 boss display names (MonsterEncounter::as_str); bump when adding acts
_BOSS_NAME_TO_IDX = {"The Guardian": 0, "Hexaghost": 1, "Slime Boss": 2}

# Flat map-global meta — bypasses the CNN/global-avg-pool that erases spatial info
ENCODING_DIM_MAP_META = (
    1                          # Floor depth (y_current / MAP_HEIGHT)
    + 1                        # On-map sentinel (y_current is not None)
    + len(_BOSS_NAME_TO_IDX)   # Act-boss identity OHE
)


def _encode_map_into(map_: Map, out: np.ndarray) -> None:
    # Room kind OHE + outgoing-edge multi-hot, per node
    for y, row in enumerate(map_.rooms):
        for x, room in enumerate(row):
            if room is None:
                continue

            idx_room_kind = _ROOM_KIND_TO_IDX[room.room_kind]
            out[y, x, idx_room_kind] = 1.0
            for x_next in room.edges:
                if 0 <= x_next < MAP_WIDTH:
                    out[y, x, len(RoomKind) + x_next] = 1.0

    # Current position
    if (
        map_.y_current is not None
        and map_.x_current is not None
    ):
        out[map_.y_current, map_.x_current, _NUM_CHANNELS - 1] = 1.0


def encode_batch_map(batch_map: list[Map], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_map)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS), dtype=np.float32)

    for b, map_ in enumerate(batch_map):
        _encode_map_into(map_, x_out[b])

    return torch.from_numpy(x_out).to(device)


def _encode_map_meta_into(map_: Map, out: np.ndarray) -> None:
    # Floor depth + on-map sentinel
    if map_.y_current is not None:
        out[0] = map_.y_current / MAP_HEIGHT
        out[1] = 1.0

    # Act-boss identity OHE
    idx_boss = _BOSS_NAME_TO_IDX.get(map_.boss_name)
    if idx_boss is not None:
        out[2 + idx_boss] = 1.0


def encode_batch_map_meta(batch_map: list[Map], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_map)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, ENCODING_DIM_MAP_META), dtype=np.float32)

    for b, map_ in enumerate(batch_map):
        _encode_map_meta_into(map_, x_out[b])

    return torch.from_numpy(x_out).to(device)
