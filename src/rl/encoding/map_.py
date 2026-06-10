import numpy as np
import torch
from slai import ChestKind
from slai import Map
from slai import RoomKind
from slai import members

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH


_ROOM_KIND_TO_IDX = {room_kind: i for i, room_kind in enumerate(members(RoomKind))}
_CHEST_KIND_TO_IDX = {chest_kind: i for i, chest_kind in enumerate(members(ChestKind))}
_NUM_CHANNELS = (
    len(_ROOM_KIND_TO_IDX)          # Room kind OHE
    + MAP_WIDTH            # Outgoing-edge multi-hot
    + 1                    # Current position
)

# Act-1 boss display names (MonsterEncounter::as_str); bump when adding acts
_BOSS_NAME_TO_IDX = {"The Guardian": 0, "Hexaghost": 1, "Slime Boss": 2}

# Flat map-global meta — position-anchored facts the pooled CNN summary can't carry
ENCODING_DIM_MAP_META = (
    1                          # Floor depth (y_current / MAP_HEIGHT)
    + 1                        # On-map sentinel (y_current is not None)
    + len(_BOSS_NAME_TO_IDX)   # Act-boss identity OHE
    + len(_ROOM_KIND_TO_IDX)   # Current room kind OHE
    + len(_CHEST_KIND_TO_IDX)  # Current chest kind OHE
    + MAP_WIDTH * len(_ROOM_KIND_TO_IDX)  # Next-row room kind OHE per column
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
                    out[y, x, len(_ROOM_KIND_TO_IDX) + x_next] = 1.0

    # Current position (the act boss sits off-grid at y_current == MAP_HEIGHT; skip it)
    if (
        map_.y_current is not None
        and map_.x_current is not None
        and map_.y_current < MAP_HEIGHT
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

    # Current room + chest kind OHE.
    # Skip the off-grid boss row (y_current == MAP_HEIGHT); act-boss identity above covers it.
    if map_.y_current is not None and map_.y_current < MAP_HEIGHT:
        room = map_.rooms[map_.y_current][map_.x_current]
        if room is not None:
            out[2 + len(_BOSS_NAME_TO_IDX) + _ROOM_KIND_TO_IDX[room.room_kind]] = 1.0
            if room.chest_kind is not None:
                base = 2 + len(_BOSS_NAME_TO_IDX) + len(_ROOM_KIND_TO_IDX)
                out[base + _CHEST_KIND_TO_IDX[room.chest_kind]] = 1.0

    # Next-row room kinds per column — the RoomSelect candidates (row 0 before the
    # first pick; nothing past the top row, where the off-grid act boss is next).
    y_next = 0 if map_.y_current is None else map_.y_current + 1
    if y_next < MAP_HEIGHT:
        base = 2 + len(_BOSS_NAME_TO_IDX) + len(_ROOM_KIND_TO_IDX) + len(_CHEST_KIND_TO_IDX)
        for x, room in enumerate(map_.rooms[y_next]):
            if room is not None:
                out[base + x * len(_ROOM_KIND_TO_IDX) + _ROOM_KIND_TO_IDX[room.room_kind]] = 1.0


def encode_batch_map_meta(batch_map: list[Map], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_map)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, ENCODING_DIM_MAP_META), dtype=np.float32)

    for b, map_ in enumerate(batch_map):
        _encode_map_meta_into(map_, x_out[b])

    return torch.from_numpy(x_out).to(device)
