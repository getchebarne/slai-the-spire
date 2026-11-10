import numpy as np
from slai import GameState
from slai import Room

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.schema.map_ import EDGE_DELTAS
from src.rl.schema.map_ import MAP_EVENT_NAME
from src.rl.schema.map_ import MAP_MONSTER_ENCOUNTER_BOSS
from src.rl.schema.map_ import MAP_NUM_ROOM_KINDS
from src.rl.schema.map_ import MAP_ROOM_KIND
from src.rl.schema.map_ import NUM_CHANNELS
from src.rl.schema.map_ import OFF_MAP_ACT
from src.rl.schema.map_ import OFF_MAP_BOSS
from src.rl.schema.map_ import OFF_MAP_BOSS_FIGHTS_AFTER
from src.rl.schema.map_ import OFF_MAP_BURNING
from src.rl.schema.map_ import OFF_MAP_CURRENT_ROOM_KIND
from src.rl.schema.map_ import OFF_MAP_EDGES
from src.rl.schema.map_ import OFF_MAP_FLOOR_DEPTH
from src.rl.schema.map_ import OFF_MAP_KEYS
from src.rl.schema.map_ import OFF_MAP_NEXT_ROOM_KINDS
from src.rl.schema.map_ import OFF_MAP_ON_MAP
from src.rl.schema.map_ import OFF_MAP_POOL_EVENT_SPECIAL
from src.rl.schema.map_ import OFF_MAP_POOL_EVENTS
from src.rl.schema.map_ import OFF_MAP_POSITION
from src.rl.schema.map_ import OFF_MAP_ROOM_KIND
from src.rl.schema.map_ import OFF_MAP_ROOM_KIND_RESOLVED
from src.rl.schema.map_ import OFF_MAP_UNKNOWN_CHANCE_MONSTER
from src.rl.schema.map_ import OFF_MAP_UNKNOWN_CHANCE_SHOP
from src.rl.schema.map_ import OFF_MAP_UNKNOWN_CHANCE_TREASURE
from src.rl.tokens import get_room_entities


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

            grid[y, x, OFF_MAP_ROOM_KIND + MAP_ROOM_KIND[room.room_kind]] = 1
            for x_next in room.edges:
                delta = x_next - x  # -1, 0, or 1
                if 0 <= x_next < MAP_WIDTH and delta in EDGE_DELTAS:
                    grid[y, x, OFF_MAP_EDGES + EDGE_DELTAS.index(delta)] = 1

            if room.burning:
                grid[y, x, OFF_MAP_BURNING] = 1

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
        out[OFF_MAP_FLOOR_DEPTH] = y_current / MAP_HEIGHT
        out[OFF_MAP_ON_MAP] = 1.0

    # Act-boss identity OHE — fail loudly on a boss we haven't mapped (new act)
    idx_boss = MAP_MONSTER_ENCOUNTER_BOSS.get(boss)
    if idx_boss is None:
        raise ValueError(f"Unknown act boss {boss!r}; add it to MAP_MONSTER_ENCOUNTER_BOSS")

    out[OFF_MAP_BOSS + idx_boss] = 1.0
    out[OFF_MAP_BOSS_FIGHTS_AFTER] = boss_fights_after

    # Current `RoomKind` OHE
    if y_current is not None and y_current < MAP_HEIGHT:
        room = rooms[y_current][x_current]
        if room is not None:
            out[OFF_MAP_CURRENT_ROOM_KIND + MAP_ROOM_KIND[room.room_kind]] = 1.0

    # Next-row `RoomKind`s per column
    for room in get_room_entities(game_state):
        idx_room_kind = MAP_ROOM_KIND[room.room_kind]
        out[OFF_MAP_NEXT_ROOM_KINDS + room.x * MAP_NUM_ROOM_KINDS + idx_room_kind] = 1.0

    # Act OHE (1-based)
    out[OFF_MAP_ACT + (act - 1)] = 1.0
    _encode_keys_into(game_state, out)
    _encode_unknown_room_into(game_state, out)


def _encode_keys_into(game_state: GameState, out: np.ndarray) -> None:
    keys = (game_state.key_emerald, game_state.key_ruby, game_state.key_sapphire)
    out[OFF_MAP_KEYS : OFF_MAP_KEYS + len(keys)] = keys


def _encode_unknown_room_into(game_state: GameState, out: np.ndarray) -> None:
    # The chances drift up by their base until rolled, and the roll caps them at 1
    out[OFF_MAP_UNKNOWN_CHANCE_MONSTER] = min(game_state.unknown_chance_monster, 1.0)
    out[OFF_MAP_UNKNOWN_CHANCE_SHOP] = min(game_state.unknown_chance_shop, 1.0)
    out[OFF_MAP_UNKNOWN_CHANCE_TREASURE] = min(game_state.unknown_chance_treasure, 1.0)

    # None before the first Room
    room_kind_resolved = game_state.map.room_kind_resolved
    if room_kind_resolved is not None:
        out[OFF_MAP_ROOM_KIND_RESOLVED + MAP_ROOM_KIND[room_kind_resolved]] = 1.0

    for event_name in game_state.pool_events:
        out[OFF_MAP_POOL_EVENTS + MAP_EVENT_NAME[event_name]] = 1.0
    for event_name in game_state.pool_event_special:
        out[OFF_MAP_POOL_EVENT_SPECIAL + MAP_EVENT_NAME[event_name]] = 1.0


def encode_map_grid_into(game_state: GameState, out: np.ndarray) -> None:
    """Copy the cached static grid and stamp the current position into a state's grid."""
    map_ = game_state.map
    out[:] = _get_static_grid(map_.identity_hash, map_.rooms)
    y_current, x_current = map_.y_current, map_.x_current
    if y_current is not None and x_current is not None and y_current < MAP_HEIGHT:
        out[y_current, x_current, OFF_MAP_POSITION] = 1
