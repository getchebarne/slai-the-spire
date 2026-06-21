import numpy as np
import torch
from slai import Map
from slai import MonsterEncounter
from slai import RoomKind
from slai import members

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.types import Slice
from src.rl.types import SliceKind

# Order = fill order = Core's global-offset order
SLICE_ROOMS = [Slice(SliceKind.ROOMS, MAP_WIDTH)]

_MAP_ROOM_KIND = {room_kind: i for i, room_kind in enumerate(members(RoomKind))}
_MAP_BOSS = {
    MonsterEncounter.TheGuardian: 0,
    MonsterEncounter.Hexaghost: 1,
    MonsterEncounter.SlimeBoss: 2,
}
MAP_NUM_ROOM_KINDS = len(_MAP_ROOM_KIND)  # per-column key-side feature dim (item 9)
# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant.
NUM_EDGE_CHANNELS = 3
NUM_CHANNELS = (
    MAP_NUM_ROOM_KINDS  # Room kind OHE
    + NUM_EDGE_CHANNELS  # Outgoing-edge multi-hot (relative {-1, 0, +1})
    + 1  # Current position
)

# Flat map-global meta — position-anchored facts the pooled CNN summary can't carry
ENCODING_DIM_MAP_META = (
    1  # Floor depth (y_current / MAP_HEIGHT)
    + 1  # On-map sentinel (y_current is not None)
    + len(_MAP_BOSS)  # Act-boss identity OHE
    + len(_MAP_ROOM_KIND)  # Current room kind OHE
    + MAP_WIDTH * len(_MAP_ROOM_KIND)  # Next-row room kind OHE per column
)


def _encode_map_into(map_: Map, out: np.ndarray) -> None:
    # Room kind OHE + outgoing-edge multi-hot, per node
    for y, row in enumerate(map_.rooms):
        for x, room in enumerate(row):
            if room is None:
                continue

            idx_room_kind = _MAP_ROOM_KIND[room.room_kind]
            out[y, x, idx_room_kind] = 1.0
            for x_next in room.edges:
                delta = x_next - x  # engine edges stay within ±1 column
                if 0 <= x_next < MAP_WIDTH and -1 <= delta <= 1:
                    out[y, x, MAP_NUM_ROOM_KINDS + delta + 1] = 1.0

    # Current position (the act boss sits off-grid at y_current == MAP_HEIGHT; skip it)
    if map_.y_current is not None and map_.x_current is not None and map_.y_current < MAP_HEIGHT:
        out[map_.y_current, map_.x_current, NUM_CHANNELS - 1] = 1.0


def _encode_map_meta_into(map_: Map, out: np.ndarray) -> None:
    # Floor depth + on-map sentinel
    if map_.y_current is not None:
        out[0] = map_.y_current / MAP_HEIGHT
        out[1] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = _MAP_BOSS.get(map_.boss)
    if idx_boss is None:
        raise ValueError(f"Unknown act boss {map_.boss!r}; add it to _MAP_BOSS")
    out[2 + idx_boss] = 1.0

    # Current room kind OHE; skip the off-grid boss row (act-boss identity covers it).
    if map_.y_current is not None and map_.y_current < MAP_HEIGHT:
        room = map_.rooms[map_.y_current][map_.x_current]
        if room is not None:
            out[2 + len(_MAP_BOSS) + _MAP_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row room kinds per column — the RoomSelect candidates (row 0 before the first pick).
    y_next = 0 if map_.y_current is None else map_.y_current + 1
    if y_next < MAP_HEIGHT:
        base = 2 + len(_MAP_BOSS) + len(_MAP_ROOM_KIND)
        for x, room in enumerate(map_.rooms[y_next]):
            if room is not None:
                out[base + x * len(_MAP_ROOM_KIND) + _MAP_ROOM_KIND[room.room_kind]] = 1.0


def _room_node_idx_into(map_: Map, out: np.ndarray) -> None:
    """Per-column flattened node index (y_next*MAP_WIDTH + x) of each next-row selectable
    room, -1 where the column has no legal room. Mirrors the engine's RoomSelect
    enumeration exactly (Start → any non-None row-0 room; Overworld → edge + non-None
    next room). The map GNN gathers its room-token embeddings at these indices; -1 marks
    an invalid (masked) room slot."""
    y_next = 0 if map_.y_current is None else map_.y_current + 1
    if y_next >= MAP_HEIGHT:
        return  # next step is the off-grid act boss (RoomSelect not enumerated there)
    if map_.y_current is None:
        for x in range(MAP_WIDTH):
            if map_.rooms[0][x] is not None:
                out[x] = y_next * MAP_WIDTH + x
        return
    cur = map_.rooms[map_.y_current][map_.x_current]
    if cur is not None:
        for x_next in cur.edges:
            if 0 <= x_next < MAP_WIDTH and map_.rooms[y_next][x_next] is not None:
                out[x_next] = y_next * MAP_WIDTH + x_next


def encode_batch_map(
    batch_map: list[Map], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the map into (grid, room_node_idx, meta): the GNN node-feature grid, the per-column
    next-row room node indices (gather + mask), and the flat position-anchored meta vector."""
    batch_size = len(batch_map)
    np_grid = np.zeros((batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.float32)
    np_node_idx = np.full((batch_size, MAP_WIDTH), -1, dtype=np.int64)
    np_meta = np.zeros((batch_size, ENCODING_DIM_MAP_META), dtype=np.float32)

    for b, map_ in enumerate(batch_map):
        _encode_map_into(map_, np_grid[b])
        _room_node_idx_into(map_, np_node_idx[b])
        _encode_map_meta_into(map_, np_meta[b])

    return (
        torch.from_numpy(np_grid).to(device),
        torch.from_numpy(np_node_idx).to(device),
        torch.from_numpy(np_meta).to(device),
    )
