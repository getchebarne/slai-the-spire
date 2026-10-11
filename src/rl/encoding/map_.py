from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import EventName
from slai import GameEnv
from slai import GameState
from slai import MonsterEncounter
from slai import RelicName
from slai import Room
from slai import RoomKind

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator


_MAP_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}
_MAP_MONSTER_ENCOUNTER_BOSS = {
    MonsterEncounter.TheGuardian: 0,
    MonsterEncounter.Hexaghost: 1,
    MonsterEncounter.SlimeBoss: 2,
    MonsterEncounter.BronzeAutomaton: 3,
    MonsterEncounter.TheCollector: 4,
    MonsterEncounter.Champ: 5,
}
_MAP_NUM_ROOM_KINDS = len(_MAP_ROOM_KIND)
_MAP_EVENT_NAME = {event_name: int(event_name) for event_name in EventName.members()}

# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant
NUM_EDGE_CHANNELS = 3


class MapGridChannel(StrEnum):
    ROOM_KIND = auto()
    EDGES = auto()
    POSITION = auto()


CHANNEL_LAYOUT_MAP_GRID, _alloc = make_channel_allocator()
OFF_MAP_ROOM_KIND = _alloc(MapGridChannel.ROOM_KIND, _MAP_NUM_ROOM_KINDS)
OFF_MAP_EDGES = _alloc(MapGridChannel.EDGES, NUM_EDGE_CHANNELS)
OFF_MAP_POSITION = _alloc(MapGridChannel.POSITION)

NUM_CHANNELS = sum(w for _, w in CHANNEL_LAYOUT_MAP_GRID)

# GNN-facing view of the edge block (read by `map_encoder.py`)
SLICE_MAP_EDGES = slice(OFF_MAP_EDGES, OFF_MAP_EDGES + NUM_EDGE_CHANNELS)


# Flat map-global meta — position-anchored facts the pooled CNN summary can't carry
class MapMetaChannel(StrEnum):
    FLOOR_DEPTH = auto()  # y_current / MAP_HEIGHT
    ON_MAP = auto()  # y_current is not None
    BOSS = auto()  # act-boss identity OHE
    CURRENT_ROOM_KIND = auto()
    NEXT_ROOM_KINDS = auto()  # RoomKind OHE per next-row column
    ACT = auto()  # each act is a fresh 15-row map; depth alone is ambiguous

    # "?" Rooms
    UNKNOWN_CHANCE_MONSTER = auto()  # the event chance is what the three odds leave
    UNKNOWN_CHANCE_SHOP = auto()
    UNKNOWN_CHANCE_TREASURE = auto()
    ROOM_KIND_RESOLVED = auto()  # a "?" stays Unknown in `rooms`; no Shop right after a Shop
    POOL_EVENTS = auto()  # events a "?" can still become, each drawn once
    POOL_EVENT_SPECIAL = auto()


CHANNEL_LAYOUT_MAP_META, _alloc = make_channel_allocator()
OFF_MAP_FLOOR_DEPTH = _alloc(MapMetaChannel.FLOOR_DEPTH)
OFF_MAP_ON_MAP = _alloc(MapMetaChannel.ON_MAP)
OFF_MAP_BOSS = _alloc(MapMetaChannel.BOSS, len(_MAP_MONSTER_ENCOUNTER_BOSS))
OFF_MAP_CURRENT_ROOM_KIND = _alloc(MapMetaChannel.CURRENT_ROOM_KIND, _MAP_NUM_ROOM_KINDS)
OFF_MAP_NEXT_ROOM_KINDS = _alloc(MapMetaChannel.NEXT_ROOM_KINDS, MAP_WIDTH * _MAP_NUM_ROOM_KINDS)
OFF_MAP_ACT = _alloc(MapMetaChannel.ACT, GameEnv.ACT_FINAL)
OFF_MAP_UNKNOWN_CHANCE_MONSTER = _alloc(MapMetaChannel.UNKNOWN_CHANCE_MONSTER)
OFF_MAP_UNKNOWN_CHANCE_SHOP = _alloc(MapMetaChannel.UNKNOWN_CHANCE_SHOP)
OFF_MAP_UNKNOWN_CHANCE_TREASURE = _alloc(MapMetaChannel.UNKNOWN_CHANCE_TREASURE)
OFF_MAP_ROOM_KIND_RESOLVED = _alloc(MapMetaChannel.ROOM_KIND_RESOLVED, _MAP_NUM_ROOM_KINDS)
OFF_MAP_POOL_EVENTS = _alloc(MapMetaChannel.POOL_EVENTS, len(_MAP_EVENT_NAME))
OFF_MAP_POOL_EVENT_SPECIAL = _alloc(MapMetaChannel.POOL_EVENT_SPECIAL, len(_MAP_EVENT_NAME))

ENCODING_DIM_MAP_META = sum(w for _, w in CHANNEL_LAYOUT_MAP_META)

# Static map-grid cache keyed by map.identity_hash: room kinds + edges are fixed per map
_MAP_GRID_CACHE: dict[int, np.ndarray] = {}
_MAP_GRID_CACHE_MAX = 100_000


def _get_static_grid(identity_hash: int, rooms: list[list[Room | None]]) -> np.ndarray:
    grid = _MAP_GRID_CACHE.get(identity_hash)
    if grid is not None:
        return grid

    # Initialize empty grid
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.float32)

    # Iterate over rooms
    for y, row in enumerate(rooms):
        for x, room in enumerate(row):
            if room is None:
                continue

            grid[y, x, OFF_MAP_ROOM_KIND + _MAP_ROOM_KIND[room.room_kind]] = 1.0
            for x_next in room.edges:
                delta = x_next - x  # -1, 0, or 1
                if 0 <= x_next < MAP_WIDTH and -1 <= delta <= 1:
                    grid[y, x, OFF_MAP_EDGES + delta + 1] = 1.0

    # Store in the cache. Guard the master copy against future writes
    grid.flags.writeable = False
    if len(_MAP_GRID_CACHE) >= _MAP_GRID_CACHE_MAX:
        _MAP_GRID_CACHE.clear()

    _MAP_GRID_CACHE[identity_hash] = grid
    return grid


def _wing_boots_charged(state: GameState) -> bool:
    """Wing Boots with charges left: the engine offers every next-row Room, not just the edges."""
    return any(r.name == RelicName.WingBoots and r.counter > 0 for r in state.relics)


def _next_row_candidate_cols(
    rooms: list[list[Room | None]], y_current: int | None, x_current: int | None, winged: bool
) -> list[int]:
    y_next = 0 if y_current is None else y_current + 1
    if y_next >= MAP_HEIGHT:
        return []

    if y_current is None or winged:
        return [x for x in range(MAP_WIDTH) if rooms[y_next][x] is not None]

    # Agent always stands on a real room; cur is never None here, so fail loud if it is
    room_curr = rooms[y_current][x_current]
    if room_curr is None:
        raise ValueError(f"Current room ({y_current}, {x_current}) is None")

    for x_next in room_curr.edges:
        if not (0 <= x_next < MAP_WIDTH and rooms[y_next][x_next] is not None):
            raise ValueError(f"Edge ({y_current}, {x_current})->({y_next}, {x_next}) has no room")

    return list(room_curr.edges)


def _encode_map_meta_into(
    rooms: list[list[Room | None]],
    y_current: int | None,
    x_current: int | None,
    boss: MonsterEncounter,
    act: int,
    winged: bool,
    out: np.ndarray,
) -> None:
    # Floor depth + on-map sentinel
    if y_current is not None:
        out[OFF_MAP_FLOOR_DEPTH] = y_current / MAP_HEIGHT
        out[OFF_MAP_ON_MAP] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = _MAP_MONSTER_ENCOUNTER_BOSS.get(boss)
    if idx_boss is None:
        raise ValueError(f"Unknown act boss {boss!r}; add it to _MAP_MONSTER_ENCOUNTER_BOSS")

    out[OFF_MAP_BOSS + idx_boss] = 1.0

    # Current `RoomKind` OHE
    if y_current is not None and y_current < MAP_HEIGHT:
        room = rooms[y_current][x_current]
        if room is not None:
            out[OFF_MAP_CURRENT_ROOM_KIND + _MAP_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row `RoomKind`s per column
    y_next = 0 if y_current is None else y_current + 1
    for x in _next_row_candidate_cols(rooms, y_current, x_current, winged):
        idx_kind = _MAP_ROOM_KIND[rooms[y_next][x].room_kind]
        out[OFF_MAP_NEXT_ROOM_KINDS + x * _MAP_NUM_ROOM_KINDS + idx_kind] = 1.0

    # Act OHE (1-based)
    out[OFF_MAP_ACT + (act - 1)] = 1.0


def _encode_unknown_room_into(
    game_state: GameState, room_kind_resolved: RoomKind | None, out: np.ndarray
) -> None:
    # The chances drift up by their base until rolled, and the roll caps them at 1
    out[OFF_MAP_UNKNOWN_CHANCE_MONSTER] = min(game_state.unknown_chance_monster, 1.0)
    out[OFF_MAP_UNKNOWN_CHANCE_SHOP] = min(game_state.unknown_chance_shop, 1.0)
    out[OFF_MAP_UNKNOWN_CHANCE_TREASURE] = min(game_state.unknown_chance_treasure, 1.0)

    # None before the first Room
    if room_kind_resolved is not None:
        out[OFF_MAP_ROOM_KIND_RESOLVED + _MAP_ROOM_KIND[room_kind_resolved]] = 1.0

    for event_name in game_state.pool_events:
        out[OFF_MAP_POOL_EVENTS + _MAP_EVENT_NAME[event_name]] = 1.0
    for event_name in game_state.pool_event_special:
        out[OFF_MAP_POOL_EVENT_SPECIAL + _MAP_EVENT_NAME[event_name]] = 1.0


def _encode_room_node_idx_into(
    rooms: list[list[Room | None]],
    y_current: int | None,
    x_current: int | None,
    winged: bool,
    out: np.ndarray,
) -> None:
    y_next = 0 if y_current is None else y_current + 1
    for x in _next_row_candidate_cols(rooms, y_current, x_current, winged):
        out[x] = y_next * MAP_WIDTH + x


def encode_batch_map(
    game_states: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    batch_size = len(game_states)
    np_grid = np.zeros((batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.float32)
    np_node_idx = np.full((batch_size, MAP_WIDTH), -1, dtype=np.int64)
    np_meta = np.zeros((batch_size, ENCODING_DIM_MAP_META), dtype=np.float32)

    for b, game_state in enumerate(game_states):
        map_ = game_state.map
        # `map_.rooms` reconstructs the whole grid across the FFI (~23us); fetch it once per state
        rooms = map_.rooms
        y_current = map_.y_current
        x_current = map_.x_current

        # Grid: copy the cached static (room+edge) grid, then set the live position bit
        np_grid[b] = _get_static_grid(map_.identity_hash, rooms)

        # Stamp current position. Do it outside the cache to maximize cache hits
        if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
            np_grid[b, y_current, x_current, OFF_MAP_POSITION] = 1.0

        winged = _wing_boots_charged(game_state)
        _encode_room_node_idx_into(rooms, y_current, x_current, winged, np_node_idx[b])
        _encode_map_meta_into(
            rooms, y_current, x_current, map_.boss, game_state.act, winged, np_meta[b]
        )
        _encode_unknown_room_into(game_state, map_.room_kind_resolved, np_meta[b])

    # The room tokens' kind row: ROOMS at each reachable next-row column, -1 where none
    np_room_idxs = np.where(np_node_idx >= 0, SliceKind.ROOMS, -1).astype(np.int16)

    return (
        torch.from_numpy(np_grid).to(device),
        torch.from_numpy(np_node_idx).to(device),
        np_room_idxs,
        torch.from_numpy(np_meta).to(device),
    )
