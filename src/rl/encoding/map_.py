import numpy as np
from slai import ACT_FINAL
from slai import MAP_HEIGHT
from slai import MAP_WIDTH
from slai import NUM_ENCOUNTERS_EASY_ACT1
from slai import EventName
from slai import GameState
from slai import MonsterEncounter
from slai import Room
from slai import RoomKind

from src.rl.encoding.layout import make_channel_allocator
from src.rl.tokens import get_burning_path
from src.rl.tokens import get_entities
from src.rl.tokens import get_room_entities
from src.rl.tokens import get_rooms
from src.rl.tokens import get_y_x_current
from src.rl.tokens import next_row_candidate_cols


_IDX_OF_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}
_IDX_OF_MONSTER_ENCOUNTER_BOSS = {
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
_MAP_NUM_ROOM_KINDS = len(_IDX_OF_ROOM_KIND)
_IDX_OF_EVENT_NAME = {event_name: int(event_name) for event_name in EventName.members()}

# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant
EDGE_DELTAS = (-1, 0, 1)


# Every channel of a cell must be 0 or 1 so Map dedup can pack each feature into one bit
_CHANNEL_LAYOUT_MAP_GRID, _alloc = make_channel_allocator()
_OFF_MAP_ROOM_KIND = _alloc("room_kind", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_EDGES = _alloc("edges", len(EDGE_DELTAS))
_OFF_MAP_BURNING = _alloc("burning")
OFF_MAP_POSITION = _alloc("position")
_OFF_MAP_BURNING_PATH = _alloc("burning_path")
_OFF_MAP_NEXT_ROOM = _alloc("next_room")

# The channels above are the Map network's cell features; the one below names the Room tokens'
# cells
NUM_FEATURE_CHANNELS = sum(w for _, w in _CHANNEL_LAYOUT_MAP_GRID)
OFF_MAP_SELECTABLE = _alloc("selectable")

NUM_CHANNELS = sum(w for _, w in _CHANNEL_LAYOUT_MAP_GRID)

# GNN-facing view of the edge block (read by `map_encoder.py`)
SLICE_MAP_EDGES = slice(_OFF_MAP_EDGES, _OFF_MAP_EDGES + len(EDGE_DELTAS))

SLICE_MAP_ROOM_KIND = slice(_OFF_MAP_ROOM_KIND, _OFF_MAP_ROOM_KIND + _MAP_NUM_ROOM_KINDS)


# Flat Map-global meta — position-anchored facts the pooled GNN readout can't carry


_CHANNEL_LAYOUT_MAP_META, _alloc = make_channel_allocator()
_OFF_MAP_FLOOR_DEPTH = _alloc("floor_depth")
_OFF_MAP_ON_MAP = _alloc("on_map")
_OFF_MAP_ENCOUNTER_BOSS = _alloc("encounter_boss", len(_IDX_OF_MONSTER_ENCOUNTER_BOSS))
_OFF_MAP_ENCOUNTER_POOL_BOSS_COUNT = _alloc("encounter_pool_boss_count")
_OFF_MAP_CURRENT_ROOM_KIND = _alloc("current_room_kind", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_NEXT_ROOM_KINDS = _alloc("next_room_kinds", MAP_WIDTH * _MAP_NUM_ROOM_KINDS)
_OFF_MAP_ACT = _alloc("act", ACT_FINAL)
_OFF_MAP_KEYS = _alloc("keys", 3)
_OFF_MAP_ENCOUNTERS_EASY_LEFT = _alloc("encounters_easy_left")
# "?" Rooms
_OFF_MAP_UNKNOWN_CHANCE_MONSTER = _alloc("unknown_chance_monster")
_OFF_MAP_UNKNOWN_CHANCE_SHOP = _alloc("unknown_chance_shop")
_OFF_MAP_UNKNOWN_CHANCE_TREASURE = _alloc("unknown_chance_treasure")
_OFF_MAP_ROOM_KIND_RESOLVED = _alloc("room_kind_resolved", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_POOL_EVENTS = _alloc("pool_events", len(_IDX_OF_EVENT_NAME))
_OFF_MAP_POOL_EVENT_SPECIAL = _alloc("pool_event_special", len(_IDX_OF_EVENT_NAME))

ENCODING_DIM_MAP_META = sum(w for _, w in _CHANNEL_LAYOUT_MAP_META)

# Static Map-grid cache keyed by map.identity_hash: Room kinds + edges + burning are fixed per Map
_MAP_GRID_CACHE: dict[int, np.ndarray] = {}
_MAP_GRID_CACHE_MAX = 100_000


def _get_static_grid(game_state: GameState) -> np.ndarray:
    identity_hash = game_state.map.identity_hash
    np_grid = _MAP_GRID_CACHE.get(identity_hash)
    if np_grid is not None:
        return np_grid

    # Initialize empty grid
    np_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.uint8)

    # Iterate over Rooms
    rooms = get_rooms(game_state)
    for y, row in enumerate(rooms):
        for x, room in enumerate(row):
            if room is None:
                continue

            np_grid[y, x, _OFF_MAP_ROOM_KIND + _IDX_OF_ROOM_KIND[room.room_kind]] = 1
            for x_next in room.edges:
                delta = x_next - x  # -1, 0, or 1
                if 0 <= x_next < MAP_WIDTH and delta in EDGE_DELTAS:
                    if y + 1 >= MAP_HEIGHT or rooms[y + 1][x_next] is None:
                        raise ValueError(f"Edge ({y}, {x})->({y + 1}, {x_next}) has no room")

                    np_grid[y, x, _OFF_MAP_EDGES + EDGE_DELTAS.index(delta)] = 1

            if room.burning:
                np_grid[y, x, _OFF_MAP_BURNING] = 1

    # Store in the cache. Guard the master copy against future writes
    np_grid.flags.writeable = False
    if len(_MAP_GRID_CACHE) >= _MAP_GRID_CACHE_MAX:
        _MAP_GRID_CACHE.clear()

    _MAP_GRID_CACHE[identity_hash] = np_grid

    return np_grid


def encode_map_meta_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write map position, boss, keys, room probabilities and event-pool features.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_MAP_META,) float32. The context row's Map range, written in place.
    """
    map_ = game_state.map
    y_current, x_current = get_y_x_current(map_)
    encounter_boss = game_state.encounter_boss
    encounter_pool_boss_count = game_state.encounter_pool_boss_count
    act = game_state.act
    # Floor depth + on-map sentinel
    if y_current is not None:
        np_out[_OFF_MAP_FLOOR_DEPTH] = y_current / MAP_HEIGHT
        np_out[_OFF_MAP_ON_MAP] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = _IDX_OF_MONSTER_ENCOUNTER_BOSS.get(encounter_boss)
    if idx_boss is None:
        raise ValueError(
            f"Unknown act boss {encounter_boss!r}; add it to _IDX_OF_MONSTER_ENCOUNTER_BOSS"
        )

    np_out[_OFF_MAP_ENCOUNTER_BOSS + idx_boss] = 1.0
    np_out[_OFF_MAP_ENCOUNTER_POOL_BOSS_COUNT] = encounter_pool_boss_count

    # Current `RoomKind` OHE
    if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
        id_room = map_.id_rooms[y_current][x_current]
        if id_room is not None:
            (room,) = get_entities(game_state, [id_room], Room)
            np_out[_OFF_MAP_CURRENT_ROOM_KIND + _IDX_OF_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row `RoomKind`s per column
    for room in get_room_entities(game_state):
        idx_room_kind = _IDX_OF_ROOM_KIND[room.room_kind]
        np_out[_OFF_MAP_NEXT_ROOM_KINDS + room.x * _MAP_NUM_ROOM_KINDS + idx_room_kind] = 1.0

    # Act OHE (1-based)
    np_out[_OFF_MAP_ACT + (act - 1)] = 1.0
    _encode_keys_into(game_state, np_out)
    np_out[_OFF_MAP_ENCOUNTERS_EASY_LEFT] = (
        game_state.encounters_easy_left / NUM_ENCOUNTERS_EASY_ACT1
    )
    _encode_unknown_room_into(game_state, np_out)


def _encode_keys_into(game_state: GameState, np_out: np.ndarray) -> None:
    keys = (game_state.key_emerald, game_state.key_ruby, game_state.key_sapphire)
    np_out[_OFF_MAP_KEYS : _OFF_MAP_KEYS + len(keys)] = keys


def _encode_unknown_room_into(game_state: GameState, np_out: np.ndarray) -> None:
    # The chances drift up by their base until rolled, and the roll caps them at 1
    np_out[_OFF_MAP_UNKNOWN_CHANCE_MONSTER] = min(game_state.unknown_chance_monster, 1.0)
    np_out[_OFF_MAP_UNKNOWN_CHANCE_SHOP] = min(game_state.unknown_chance_shop, 1.0)
    np_out[_OFF_MAP_UNKNOWN_CHANCE_TREASURE] = min(game_state.unknown_chance_treasure, 1.0)

    # None before the first Room
    room_kind_resolved = game_state.map.room_kind_resolved
    if room_kind_resolved is not None:
        np_out[_OFF_MAP_ROOM_KIND_RESOLVED + _IDX_OF_ROOM_KIND[room_kind_resolved]] = 1.0

    for event_name in game_state.pool_events:
        np_out[_OFF_MAP_POOL_EVENTS + _IDX_OF_EVENT_NAME[event_name]] = 1.0
    for event_name in game_state.pool_event_special:
        np_out[_OFF_MAP_POOL_EVENT_SPECIAL + _IDX_OF_EVENT_NAME[event_name]] = 1.0


def encode_map_grid_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Copy the cached static grid; stamp the position, burning path, next and selectable Rooms.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) uint8. The state's Map grid, overwritten.
    """
    map_ = game_state.map
    np_out[:] = _get_static_grid(game_state)
    y_current, x_current = get_y_x_current(map_)
    if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
        np_out[y_current, x_current, OFF_MAP_POSITION] = 1

    # The Rooms that still reach the burning Room while the Emerald Key is missing
    for y, x in get_burning_path(game_state):
        np_out[y, x, _OFF_MAP_BURNING_PATH] = 1

    # The next-row Rooms the position's edges reach, Wing Boots aside
    y_next = 0 if y_current is None else y_current + 1
    for x_next in next_row_candidate_cols(game_state, y_current, x_current, winged=False):
        np_out[y_next, x_next, _OFF_MAP_NEXT_ROOM] = 1

    # The Room tokens' cells, in the tokens' order: cell order
    for room in get_room_entities(game_state):
        np_out[room.y, room.x, OFF_MAP_SELECTABLE] = 1
