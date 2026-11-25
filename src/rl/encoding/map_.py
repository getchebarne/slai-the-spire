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

from src.rl.constants import EDGE_DELTAS
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.room import ENCODING_DIM_ROOM
from src.rl.encoding.room import encode_room_into
from src.rl.tokens import get_burning_path
from src.rl.tokens import get_cells_next_room
from src.rl.tokens import get_entities
from src.rl.tokens import get_map_key
from src.rl.tokens import get_room_entities
from src.rl.tokens import get_rooms_forward
from src.rl.tokens import get_y_x_current
from src.rl.types import MapKey
from src.rl.utils import get_edge_channels


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


# A cell's topology: a forward Room sits there, its edges, a RoomSelect can name it
_CHANNEL_LAYOUT_MAP_TOPOLOGY, _alloc = make_channel_allocator()
OFF_MAP_ROOM_FORWARD = _alloc("room_forward")
_OFF_MAP_EDGES = _alloc("edges", len(EDGE_DELTAS))
OFF_MAP_SELECTABLE = _alloc("selectable")

MAP_TOPOLOGY_CHANNELS = sum(w for _, w in _CHANNEL_LAYOUT_MAP_TOPOLOGY)

# GNN-facing view of the edge block (read by `map_encoder.py`)
SLICE_MAP_EDGES = slice(_OFF_MAP_EDGES, _OFF_MAP_EDGES + len(EDGE_DELTAS))

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
_OFF_MAP_UNKNOWN_CHANCE_MONSTER = _alloc("unknown_chance_monster")
_OFF_MAP_UNKNOWN_CHANCE_SHOP = _alloc("unknown_chance_shop")
_OFF_MAP_UNKNOWN_CHANCE_TREASURE = _alloc("unknown_chance_treasure")
_OFF_MAP_ROOM_KIND_RESOLVED = _alloc("room_kind_resolved", _MAP_NUM_ROOM_KINDS)
_OFF_MAP_POOL_EVENTS = _alloc("pool_events", len(_IDX_OF_EVENT_NAME))
_OFF_MAP_POOL_EVENT_SPECIAL = _alloc("pool_event_special", len(_IDX_OF_EVENT_NAME))

ENCODING_DIM_MAP_META = sum(w for _, w in _CHANNEL_LAYOUT_MAP_META)

# Map cache keyed by `get_map_key`: a Map's features and its topology
_MAP_OF_MAP_KEY: dict[MapKey, tuple[np.ndarray, np.ndarray]] = {}
_MAP_CACHE_MAX = 4_096


def _get_map(game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """Find the Map of the states that share a Map key, making it the first time it comes.

    Parameters
    ----------
    game_state : GameState
        The state to encode.

    Returns
    -------
    np_feature : np.ndarray
        (MAP_HEIGHT, MAP_WIDTH, ENCODING_DIM_ROOM) uint8. Each forward Room's feature row at its
        cell, zero where the cell holds none.
    np_topology : np.ndarray
        (MAP_HEIGHT, MAP_WIDTH, MAP_TOPOLOGY_CHANNELS) uint8. Each forward Room's topology at its
        cell, zero where the cell holds none.
    """
    map_key = get_map_key(game_state)
    map_cached = _MAP_OF_MAP_KEY.get(map_key)
    if map_cached is not None:
        return map_cached

    # Initialize the empty Map
    np_feature = np.zeros((MAP_HEIGHT, MAP_WIDTH, ENCODING_DIM_ROOM), dtype=np.uint8)
    np_topology = np.zeros((MAP_HEIGHT, MAP_WIDTH, MAP_TOPOLOGY_CHANNELS), dtype=np.uint8)

    # Iterate over the forward Rooms
    rooms_forward = get_rooms_forward(game_state)
    cells_room_forward = {(room.y, room.x) for room in rooms_forward}
    burning_path = get_burning_path(game_state)
    cells_next_room = get_cells_next_room(game_state)
    for room in rooms_forward:
        y, x = room.y, room.x
        edge_channels = get_edge_channels(room)

        # Every edge must lead to a forward Room's cell
        for delta, edge in zip(EDGE_DELTAS, edge_channels):
            if edge and (y + 1, x + delta) not in cells_room_forward:
                raise ValueError(f"Edge ({y}, {x})->({y + 1}, {x + delta}) has no forward Room")

        encode_room_into(room, (y, x) in burning_path, (y, x) in cells_next_room, np_feature[y, x])
        np_topology[y, x, OFF_MAP_ROOM_FORWARD] = 1
        np_topology[y, x, SLICE_MAP_EDGES] = edge_channels

    # The Room tokens' cells, in the tokens' order: cell order
    for room in get_room_entities(game_state):
        np_topology[room.y, room.x, OFF_MAP_SELECTABLE] = 1

    # Store in the cache. Guard the master copies against future writes
    np_feature.flags.writeable = False
    np_topology.flags.writeable = False
    if len(_MAP_OF_MAP_KEY) >= _MAP_CACHE_MAX:
        _MAP_OF_MAP_KEY.clear()

    _MAP_OF_MAP_KEY[map_key] = (np_feature, np_topology)

    return np_feature, np_topology


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


def encode_map_into(
    game_state: GameState, np_feature: np.ndarray, np_topology: np.ndarray
) -> None:
    """Copy the state's Map: each forward Room's feature row and topology at its cell.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_feature : np.ndarray
        (MAP_HEIGHT, MAP_WIDTH, ENCODING_DIM_ROOM) uint8. The state's Map features, overwritten.
    np_topology : np.ndarray
        (MAP_HEIGHT, MAP_WIDTH, MAP_TOPOLOGY_CHANNELS) uint8. The state's Map topology,
        overwritten.
    """
    np_feature[:], np_topology[:] = _get_map(game_state)
