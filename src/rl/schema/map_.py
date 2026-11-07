"""Feature layout and static catalogs for map rows."""

from slai import EventName
from slai import GameEnv
from slai import MonsterEncounter
from slai import RoomKind

from src.rl.constants import MAP_WIDTH
from src.rl.schema.layout import make_channel_allocator


MAP_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}
MAP_MONSTER_ENCOUNTER_BOSS = {
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
MAP_NUM_ROOM_KINDS = len(MAP_ROOM_KIND)
MAP_EVENT_NAME = {event_name: int(event_name) for event_name in EventName.members()}

# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant
EDGE_DELTAS = (-1, 0, 1)
NUM_EDGE_CHANNELS = len(EDGE_DELTAS)


# Every cell must be 0 or 1: `map_encoder.py`'s dedup packs the grid one bit per cell
CHANNEL_LAYOUT_MAP_GRID, _alloc = make_channel_allocator()
OFF_MAP_ROOM_KIND = _alloc("room_kind", MAP_NUM_ROOM_KINDS)
OFF_MAP_EDGES = _alloc("edges", NUM_EDGE_CHANNELS)
OFF_MAP_BURNING = _alloc("burning")
OFF_MAP_POSITION = _alloc("position")

NUM_CHANNELS = sum(w for _, w in CHANNEL_LAYOUT_MAP_GRID)

# GNN-facing view of the edge block (read by `map_encoder.py`)
SLICE_MAP_EDGES = slice(OFF_MAP_EDGES, OFF_MAP_EDGES + NUM_EDGE_CHANNELS)

SLICE_MAP_ROOM_KIND = slice(OFF_MAP_ROOM_KIND, OFF_MAP_ROOM_KIND + MAP_NUM_ROOM_KINDS)


# Flat Map-global meta — position-anchored facts the pooled GNN readout can't carry


CHANNEL_LAYOUT_MAP_META, _alloc = make_channel_allocator()
OFF_MAP_FLOOR_DEPTH = _alloc("floor_depth")  # y_current / MAP_HEIGHT
OFF_MAP_ON_MAP = _alloc("on_map")  # y_current is not None
OFF_MAP_BOSS = _alloc("boss", len(MAP_MONSTER_ENCOUNTER_BOSS))  # Act-boss identity OHE
OFF_MAP_BOSS_FIGHTS_AFTER = _alloc(
    "boss_fights_after"
)  # Boss fights after the boss's: 1 at A20 in Act 3 until the second
OFF_MAP_CURRENT_ROOM_KIND = _alloc("current_room_kind", MAP_NUM_ROOM_KINDS)
OFF_MAP_NEXT_ROOM_KINDS = _alloc(
    "next_room_kinds", MAP_WIDTH * MAP_NUM_ROOM_KINDS
)  # RoomKind OHE per next-row column
OFF_MAP_ACT = _alloc(
    "act", GameEnv.ACT_FINAL
)  # Each act is a fresh 15-row Map; depth alone is ambiguous
OFF_MAP_KEYS = _alloc("keys", 3)  # The keys held: Emerald, Ruby, Sapphire
# "?" Rooms
OFF_MAP_UNKNOWN_CHANCE_MONSTER = _alloc(
    "unknown_chance_monster"
)  # The Event chance is what the three odds leave
OFF_MAP_UNKNOWN_CHANCE_SHOP = _alloc("unknown_chance_shop")
OFF_MAP_UNKNOWN_CHANCE_TREASURE = _alloc("unknown_chance_treasure")
OFF_MAP_ROOM_KIND_RESOLVED = _alloc(
    "room_kind_resolved", MAP_NUM_ROOM_KINDS
)  # A "?" stays Unknown in `rooms`; no Shop right after a Shop
OFF_MAP_POOL_EVENTS = _alloc(
    "pool_events", len(MAP_EVENT_NAME)
)  # Events a "?" can still become, each drawn once
OFF_MAP_POOL_EVENT_SPECIAL = _alloc("pool_event_special", len(MAP_EVENT_NAME))

ENCODING_DIM_MAP_META = sum(w for _, w in CHANNEL_LAYOUT_MAP_META)
