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


# Static map-grid cache keyed by map.identity_hash: room kinds + edges are fixed per map (only the
# position bit moves), and the 105-node fill is the encode hotspot. Mirrors the card/potion caches.
_MAP_GRID_CACHE: dict[int, np.ndarray] = {}
_MAP_GRID_CACHE_MAX = 100_000


def _static_grid(identity_hash: int, rooms: list) -> np.ndarray:
    """The position-independent (room kind + edge) grid for a map, cached by identity_hash."""
    grid = _MAP_GRID_CACHE.get(identity_hash)
    if grid is None:
        grid = np.zeros((MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.float32)
        for y, row in enumerate(rooms):
            for x, room in enumerate(row):
                if room is None:
                    continue

                grid[y, x, _MAP_ROOM_KIND[room.room_kind]] = 1.0
                for x_next in room.edges:
                    delta = x_next - x  # engine edges stay within ±1 column
                    if 0 <= x_next < MAP_WIDTH and -1 <= delta <= 1:
                        grid[y, x, MAP_NUM_ROOM_KINDS + delta + 1] = 1.0
        grid.flags.writeable = False  # guard the cached master copy
        if len(_MAP_GRID_CACHE) >= _MAP_GRID_CACHE_MAX:
            _MAP_GRID_CACHE.clear()
        _MAP_GRID_CACHE[identity_hash] = grid
    return grid


def _encode_map_meta_into(
    rooms: list, y_current: int | None, x_current: int | None, boss, out: np.ndarray
) -> None:
    # Floor depth + on-map sentinel
    if y_current is not None:
        out[0] = y_current / MAP_HEIGHT
        out[1] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = _MAP_BOSS.get(boss)
    if idx_boss is None:
        raise ValueError(f"Unknown act boss {boss!r}; add it to _MAP_BOSS")
    out[2 + idx_boss] = 1.0

    # Current room kind OHE; skip the off-grid boss row (act-boss identity covers it).
    if y_current is not None and y_current < MAP_HEIGHT:
        room = rooms[y_current][x_current]
        if room is not None:
            out[2 + len(_MAP_BOSS) + _MAP_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row room kinds per column — the RoomSelect candidates (row 0 before the first pick).
    y_next = 0 if y_current is None else y_current + 1
    if y_next < MAP_HEIGHT:
        base = 2 + len(_MAP_BOSS) + len(_MAP_ROOM_KIND)
        for x, room in enumerate(rooms[y_next]):
            if room is not None:
                out[base + x * len(_MAP_ROOM_KIND) + _MAP_ROOM_KIND[room.room_kind]] = 1.0


def _room_node_idx_into(
    rooms: list, y_current: int | None, x_current: int | None, out: np.ndarray
) -> None:
    """Per-column flattened node index (y_next*MAP_WIDTH + x) of each next-row selectable
    room, -1 where the column has no legal room. Mirrors the engine's RoomSelect
    enumeration exactly (Start → any non-None row-0 room; Overworld → edge + non-None
    next room). The map GNN gathers its room-token embeddings at these indices; -1 marks
    an invalid (masked) room slot."""
    y_next = 0 if y_current is None else y_current + 1
    if y_next >= MAP_HEIGHT:
        return  # next step is the off-grid act boss (RoomSelect not enumerated there)
    if y_current is None:
        for x in range(MAP_WIDTH):
            if rooms[0][x] is not None:
                out[x] = y_next * MAP_WIDTH + x
        return
    cur = rooms[y_current][x_current]
    if cur is not None:
        for x_next in cur.edges:
            if 0 <= x_next < MAP_WIDTH and rooms[y_next][x_next] is not None:
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
        # `map_.rooms` reconstructs the whole grid across the FFI (~23us); fetch it once per state
        rooms = map_.rooms
        y_current = map_.y_current
        x_current = map_.x_current
        # Grid: copy the cached static (room+edge) grid, then set the live position bit
        np_grid[b] = _static_grid(map_.identity_hash, rooms)
        if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
            np_grid[b, y_current, x_current, NUM_CHANNELS - 1] = 1.0
        _room_node_idx_into(rooms, y_current, x_current, np_node_idx[b])
        _encode_map_meta_into(rooms, y_current, x_current, map_.boss, np_meta[b])

    return (
        torch.from_numpy(np_grid).to(device),
        torch.from_numpy(np_node_idx).to(device),
        torch.from_numpy(np_meta).to(device),
    )
