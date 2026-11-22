import numpy as np
from slai import Room
from slai import RoomKind

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import RecordKeyRoom


_IDX_OF_ROOM_KIND = {room_kind: int(room_kind) for room_kind in RoomKind.members()}

# Relative outgoing edges {-1,0,+1}: engine edges are within ±1 column; translation-equivariant
EDGE_DELTAS = (-1, 0, 1)


_CHANNEL_LAYOUT_ROOM, _alloc = make_channel_allocator()
_OFF_ROOM_KIND = _alloc("room_kind", len(_IDX_OF_ROOM_KIND))
_OFF_ROOM_EDGES = _alloc("edges", len(EDGE_DELTAS))
_OFF_ROOM_BURNING = _alloc("burning")
_OFF_ROOM_BURNING_PATH = _alloc("burning_path")
_OFF_ROOM_NEXT_ROOM = _alloc("next_room")

ENCODING_DIM_ROOM = sum(w for _, w in _CHANNEL_LAYOUT_ROOM)

# Most nonzero columns a Room row holds: one per channel, but one per edge in the edges channel
ENCODING_DIM_SPARSE_ROOM = len(_CHANNEL_LAYOUT_ROOM) - 1 + len(EDGE_DELTAS)


def encode_room_into(
    room: Room, on_burning_path: bool, is_next_room: bool, np_own: np.ndarray
) -> None:
    """Write a Room's token: its own columns.

    Every input besides the Room is in the Room's record key (`get_record_key_room`), so a new
    input goes in the key too.

    Parameters
    ----------
    room : Room
        The Room.
    on_burning_path : bool
        Whether it is on a path that still reaches the burning Room.
    is_next_room : bool
        Whether it is a next-row Room the position's edges reach, Wing Boots aside.
    np_own : np.ndarray
        (ENCODING_DIM_ROOM,) float32. The Room's own range of its token row, written in place.
    """
    np_own[_OFF_ROOM_KIND + _IDX_OF_ROOM_KIND[room.room_kind]] = 1.0
    np_own[_OFF_ROOM_EDGES : _OFF_ROOM_EDGES + len(EDGE_DELTAS)] = get_edge_channels(room)
    np_own[_OFF_ROOM_BURNING] = float(room.burning)
    np_own[_OFF_ROOM_BURNING_PATH] = float(on_burning_path)
    np_own[_OFF_ROOM_NEXT_ROOM] = float(is_next_room)


def get_record_key_room(room: Room, on_burning_path: bool, is_next_room: bool) -> RecordKeyRoom:
    """Make a Room's record key: everything its row reads.

    Parameters
    ----------
    room : Room
        The Room.
    on_burning_path : bool
        Whether it is on a path that still reaches the burning Room.
    is_next_room : bool
        Whether it is a next-row Room the position's edges reach, Wing Boots aside.

    Returns
    -------
    RecordKeyRoom
        Its Room kind, its edge channels, whether it is the burning Room, whether it is on a path
        that still reaches the burning Room, and whether it is a next-row Room the position's
        edges reach.
    """
    return (
        _IDX_OF_ROOM_KIND[room.room_kind],
        get_edge_channels(room),
        room.burning,
        on_burning_path,
        is_next_room,
    )


def get_edge_channels(room: Room) -> tuple[int, ...]:
    """Turn a Room's edges into edge channels: the Room's token and the Map grid both hold them.

    Parameters
    ----------
    room : Room
        The Room.

    Returns
    -------
    tuple[int, ...]
        One 0/1 flag per `EDGE_DELTAS`: whether an edge leads to the next row, that many columns
        over.
    """
    edges = room.edges  # Bound once: each read clones the list
    x = room.x

    return tuple(int(x + delta in edges) for delta in EDGE_DELTAS)
