import numpy as np
from slai import Room
from slai import RoomKind

from src.rl.constants import EDGE_DELTAS
from src.rl.encoding.layout import make_channel_allocator
from src.rl.utils import get_edge_channels


_IDX_OF_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}

# Every channel must be 0 or 1: the Map's hash packs each into one bit
_CHANNEL_LAYOUT_ROOM, _alloc = make_channel_allocator()
_OFF_ROOM_KIND = _alloc("room_kind", len(_IDX_OF_ROOM_KIND))
_OFF_ROOM_EDGES = _alloc("edges", len(EDGE_DELTAS))
_OFF_ROOM_BURNING = _alloc("burning")
_OFF_ROOM_BURNING_PATH = _alloc("burning_path")
_OFF_ROOM_NEXT_ROOM = _alloc("next_room")

ENCODING_DIM_ROOM = sum(w for _, w in _CHANNEL_LAYOUT_ROOM)


def encode_room_into(
    room: Room, on_burning_path: bool, is_next_room: bool, np_own: np.ndarray
) -> None:
    """Write a forward Room's feature row.

    The Map's cache keeps the rows by Map key (`tokens.get_map_key`), so all that a new input
    depends on goes in that key too.

    Parameters
    ----------
    room : Room
        The Room.
    on_burning_path : bool
        Whether it is on a path that still reaches the burning Room.
    is_next_room : bool
        Whether it is a next-row Room the position's edges reach, Wing Boots aside.
    np_own : np.ndarray
        (ENCODING_DIM_ROOM,) uint8. The Room's feature row, written in place.
    """
    np_own[_OFF_ROOM_KIND + _IDX_OF_ROOM_KIND[room.room_kind]] = 1
    np_own[_OFF_ROOM_EDGES : _OFF_ROOM_EDGES + len(EDGE_DELTAS)] = get_edge_channels(room)
    np_own[_OFF_ROOM_BURNING] = room.burning
    np_own[_OFF_ROOM_BURNING_PATH] = on_burning_path
    np_own[_OFF_ROOM_NEXT_ROOM] = is_next_room
