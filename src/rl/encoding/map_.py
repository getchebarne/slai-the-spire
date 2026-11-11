import numpy as np
from slai import EventName
from slai import GameEnv
from slai import GameState
from slai import MonsterEncounter
from slai import Room
from slai import RoomKind

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.layout import make_channel_allocator
from src.rl.tokens import get_room_entities


_MAP_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}
_MAP_MONSTER_ENCOUNTER_BOSS = {
    MonsterEncounter.TheGuardian: 0,
    MonsterEncounter.Hexaghost: 1,
    MonsterEncounter.SlimeBoss: 2,
    MonsterEncounter.BronzeAutomaton: 3,
    MonsterEncounter.TheCollector: 4,
    MonsterEncounter.Champ: 5,
    MonsterEncounter.AwakenedOne: 6,
    MonsterEncounter.DonuAndDeca: 7,
    MonsterEncounter.TimeEater: 8,
    MonsterEncounter.CorruptHeart: 9,
}
_MAP_NUM_ROOM_KINDS = len(_MAP_ROOM_KIND)
_MAP_EVENT_NAME = {event_name: int(event_name) for event_name in EventName.members()}

# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant
EDGE_DELTAS = (-1, 0, 1)
NUM_EDGE_CHANNELS = len(EDGE_DELTAS)


# Every cell must be 0 or 1: `map_encoder.py`'s dedup packs the grid one bit per cell
_CHANNEL_LAYOUT_MAP_GRID, _alloc = make_channel_allocator()
_OFF_MAP_ROOM_KIND = _alloc("room_kind", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_EDGES = _alloc("edges", NUM_EDGE_CHANNELS)
_OFF_MAP_BURNING = _alloc("burning")
OFF_MAP_POSITION = _alloc("position")

NUM_CHANNELS = sum(w for _, w in _CHANNEL_LAYOUT_MAP_GRID)

# GNN-facing view of the edge block (read by `map_encoder.py`)
SLICE_MAP_EDGES = slice(_OFF_MAP_EDGES, _OFF_MAP_EDGES + NUM_EDGE_CHANNELS)

SLICE_MAP_ROOM_KIND = slice(_OFF_MAP_ROOM_KIND, _OFF_MAP_ROOM_KIND + _MAP_NUM_ROOM_KINDS)


# Flat Map-global meta — position-anchored facts the pooled GNN readout can't carry


_CHANNEL_LAYOUT_MAP_META, _alloc = make_channel_allocator()
_OFF_MAP_FLOOR_DEPTH = _alloc("floor_depth")  # y_current / MAP_HEIGHT
_OFF_MAP_ON_MAP = _alloc("on_map")  # y_current is not None
_OFF_MAP_BOSS = _alloc("boss", len(_MAP_MONSTER_ENCOUNTER_BOSS))  # Act-boss identity OHE
_OFF_MAP_BOSS_FIGHTS_AFTER = _alloc(
    "boss_fights_after"
)  # Boss fights after the boss's: 1 at A20 in Act 3 until the second
_OFF_MAP_CURRENT_ROOM_KIND = _alloc("current_room_kind", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_NEXT_ROOM_KINDS = _alloc(
    "next_room_kinds", MAP_WIDTH * _MAP_NUM_ROOM_KINDS
)  # RoomKind OHE per next-row column
_OFF_MAP_ACT = _alloc(
    "act", GameEnv.ACT_FINAL
)  # Each act is a fresh 15-row Map; depth alone is ambiguous
_OFF_MAP_KEYS = _alloc("keys", 3)  # The keys held: Emerald, Ruby, Sapphire
# "?" Rooms
_OFF_MAP_UNKNOWN_CHANCE_MONSTER = _alloc(
    "unknown_chance_monster"
)  # The Event chance is what the three odds leave
_OFF_MAP_UNKNOWN_CHANCE_SHOP = _alloc("unknown_chance_shop")
_OFF_MAP_UNKNOWN_CHANCE_TREASURE = _alloc("unknown_chance_treasure")
_OFF_MAP_ROOM_KIND_RESOLVED = _alloc(
    "room_kind_resolved", _MAP_NUM_ROOM_KINDS
)  # A "?" stays Unknown in `rooms`; no Shop right after a Shop
_OFF_MAP_POOL_EVENTS = _alloc(
    "pool_events", len(_MAP_EVENT_NAME)
)  # Events a "?" can still become, each drawn once
_OFF_MAP_POOL_EVENT_SPECIAL = _alloc("pool_event_special", len(_MAP_EVENT_NAME))

ENCODING_DIM_MAP_META = sum(w for _, w in _CHANNEL_LAYOUT_MAP_META)

# Static Map-grid cache keyed by map.identity_hash: Room kinds + edges + burning are fixed per Map
_MAP_GRID_CACHE: dict[int, np.ndarray] = {}
_MAP_GRID_CACHE_MAX = 100_000


def _get_static_grid(identity_hash: int, rooms: list[list[Room | None]]) -> np.ndarray:
    grid = _MAP_GRID_CACHE.get(identity_hash)
    if grid is not None:
        return grid

    # Initialize empty grid
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.uint8)

    # Iterate over Rooms
    for y, row in enumerate(rooms):
        for x, room in enumerate(row):
            if room is None:
                continue

            grid[y, x, _OFF_MAP_ROOM_KIND + _MAP_ROOM_KIND[room.room_kind]] = 1
            for x_next in room.edges:
                delta = x_next - x  # -1, 0, or 1
                if 0 <= x_next < MAP_WIDTH and delta in EDGE_DELTAS:
                    grid[y, x, _OFF_MAP_EDGES + EDGE_DELTAS.index(delta)] = 1

            if room.burning:
                grid[y, x, _OFF_MAP_BURNING] = 1

    # Store in the cache. Guard the master copy against future writes
    grid.flags.writeable = False
    if len(_MAP_GRID_CACHE) >= _MAP_GRID_CACHE_MAX:
        _MAP_GRID_CACHE.clear()

    _MAP_GRID_CACHE[identity_hash] = grid
    return grid


def encode_map_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write map position, boss, keys, room probabilities and event-pool features."""
    map_ = game_state.map
    rooms = map_.rooms
    y_current, x_current = map_.y_current, map_.x_current
    boss, boss_fights_after = map_.boss, map_.boss_fights_after
    act = game_state.act
    # Floor depth + on-map sentinel
    if y_current is not None:
        out[_OFF_MAP_FLOOR_DEPTH] = y_current / MAP_HEIGHT
        out[_OFF_MAP_ON_MAP] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = _MAP_MONSTER_ENCOUNTER_BOSS.get(boss)
    if idx_boss is None:
        raise ValueError(f"Unknown act boss {boss!r}; add it to _MAP_MONSTER_ENCOUNTER_BOSS")

    out[_OFF_MAP_BOSS + idx_boss] = 1.0
    out[_OFF_MAP_BOSS_FIGHTS_AFTER] = boss_fights_after

    # Current `RoomKind` OHE
    if y_current is not None and y_current < MAP_HEIGHT:
        room = rooms[y_current][x_current]
        if room is not None:
            out[_OFF_MAP_CURRENT_ROOM_KIND + _MAP_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row `RoomKind`s per column
    for room in get_room_entities(game_state):
        idx_room_kind = _MAP_ROOM_KIND[room.room_kind]
        out[_OFF_MAP_NEXT_ROOM_KINDS + room.x * _MAP_NUM_ROOM_KINDS + idx_room_kind] = 1.0

    # Act OHE (1-based)
    out[_OFF_MAP_ACT + (act - 1)] = 1.0
    _encode_keys_into(game_state, out)
    _encode_unknown_room_into(game_state, out)


def _encode_keys_into(game_state: GameState, out: np.ndarray) -> None:
    keys = (game_state.key_emerald, game_state.key_ruby, game_state.key_sapphire)
    out[_OFF_MAP_KEYS : _OFF_MAP_KEYS + len(keys)] = keys


def _encode_unknown_room_into(game_state: GameState, out: np.ndarray) -> None:
    # The chances drift up by their base until rolled, and the roll caps them at 1
    out[_OFF_MAP_UNKNOWN_CHANCE_MONSTER] = min(game_state.unknown_chance_monster, 1.0)
    out[_OFF_MAP_UNKNOWN_CHANCE_SHOP] = min(game_state.unknown_chance_shop, 1.0)
    out[_OFF_MAP_UNKNOWN_CHANCE_TREASURE] = min(game_state.unknown_chance_treasure, 1.0)

    # None before the first Room
    room_kind_resolved = game_state.map.room_kind_resolved
    if room_kind_resolved is not None:
        out[_OFF_MAP_ROOM_KIND_RESOLVED + _MAP_ROOM_KIND[room_kind_resolved]] = 1.0

    for event_name in game_state.pool_events:
        out[_OFF_MAP_POOL_EVENTS + _MAP_EVENT_NAME[event_name]] = 1.0
    for event_name in game_state.pool_event_special:
        out[_OFF_MAP_POOL_EVENT_SPECIAL + _MAP_EVENT_NAME[event_name]] = 1.0


def encode_map_grid_into(game_state: GameState, out: np.ndarray) -> None:
    """Copy the cached static grid and stamp the current position into a state's grid."""
    map_ = game_state.map
    out[:] = _get_static_grid(map_.identity_hash, map_.rooms)
    y_current, x_current = map_.y_current, map_.x_current
    if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
        out[y_current, x_current, OFF_MAP_POSITION] = 1
