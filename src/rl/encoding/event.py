import warnings
from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Effect
from slai import Event
from slai import EventName
from slai import GameState

from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator


# Order = fill order = Core's global-offset order
SLICE_EVENTS = [Slice(SliceKind.EVENT_OPTIONS, MAX_EVENT_OPTIONS)]


_EVENT_NAME_TO_IDX = {name: int(name) for name in EventName.members()}


class EventMetaChannel(StrEnum):
    NAME = auto()
    CONSUMED = auto()
    STAGE = auto()

    # Dead Adventurer only
    FOUND_GOLD = auto()
    FOUND_NOTHING = auto()
    FOUND_RELIC = auto()


CHANNEL_LAYOUT_EVENT_META, _alloc = make_channel_allocator()
OFF_EVENT_NAME = _alloc(EventMetaChannel.NAME, len(_EVENT_NAME_TO_IDX))
OFF_EVENT_CONSUMED = _alloc(EventMetaChannel.CONSUMED)
OFF_EVENT_STAGE = _alloc(EventMetaChannel.STAGE)
OFF_EVENT_FOUND_GOLD = _alloc(EventMetaChannel.FOUND_GOLD)
OFF_EVENT_FOUND_NOTHING = _alloc(EventMetaChannel.FOUND_NOTHING)
OFF_EVENT_FOUND_RELIC = _alloc(EventMetaChannel.FOUND_RELIC)

ENCODING_DIM_EVENT_META = sum(w for _, w in CHANNEL_LAYOUT_EVENT_META)

ENCODING_DIM_EVENT_OPTION = MAX_EVENT_OPTIONS  # Slot-index OHE; Effects are separate rows

_SLOT_OHE = np.eye(MAX_EVENT_OPTIONS, dtype=np.float32)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    out[OFF_EVENT_NAME + _EVENT_NAME_TO_IDX[event.name]] = 1.0
    out[OFF_EVENT_CONSUMED] = float(event.consumed)
    out[OFF_EVENT_STAGE] = min(event.stage, EVENT_STATE_CAP) / EVENT_STATE_CAP
    out[OFF_EVENT_FOUND_GOLD] = float(event.found_gold)
    out[OFF_EVENT_FOUND_NOTHING] = float(event.found_nothing)
    out[OFF_EVENT_FOUND_RELIC] = float(event.found_relic)


def encode_batch_events(
    batch_event: list[Event | None], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch_event)

    # Pre-allocate NumPy arrays
    np_meta = np.zeros((batch_size, ENCODING_DIM_EVENT_META), dtype=np.float32)
    np_options = np.zeros(
        (batch_size, MAX_EVENT_OPTIONS, ENCODING_DIM_EVENT_OPTION), dtype=np.float32
    )
    np_pad = np.zeros((batch_size, MAX_EVENT_OPTIONS), dtype=bool)

    for b, event in enumerate(batch_event):
        if event is None:
            continue

        if len(event.options) > MAX_EVENT_OPTIONS and event.name not in _WARNED_EVENT_TRUNCATED:
            _WARNED_EVENT_TRUNCATED.add(event.name)
            warnings.warn(
                f"Event {event.name!r} has {len(event.options)} options, truncated to"
                f" encoder cap ({MAX_EVENT_OPTIONS})"
            )

        _encode_event_meta_into(event, np_meta[b])
        num_options = min(len(event.options), MAX_EVENT_OPTIONS)
        np_options[b, :num_options] = _SLOT_OHE[:num_options]
        np_pad[b, :num_options] = True

    return (
        torch.from_numpy(np_meta).to(device),
        torch.from_numpy(np_options).to(device),
        torch.from_numpy(np_pad).to(device),
    )


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, effects) for every event option. The option row itself is just a
    slot OHE now — everything it does lives in these rows."""
    event = game_state.event
    if event is None:
        return
    for i, option_effects in enumerate(event.options[:MAX_EVENT_OPTIONS]):
        if option_effects:
            yield i, option_effects
