import warnings

import numpy as np
import torch
from slai import Event
from slai import EventName
from slai import EventOption
from slai import members

from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.types import Slice
from src.rl.types import SliceKind


# Order = fill order = Core's global-offset order
SLICE_EVENTS = [Slice(SliceKind.EVENT_OPTIONS, MAX_EVENT_OPTIONS)]

_EVENT_NAME_TO_IDX = {event_name: i for i, event_name in enumerate(members(EventName))}
ENCODING_DIM_EVENT_META = len(_EVENT_NAME_TO_IDX) + 1  # Name OHE  & state scalar
ENCODING_DIM_EVENT_OPTION = (
    MAX_EVENT_OPTIONS  # Slot-index OHE
    + 1  # Gated out
    + ENCODING_DIM_EFFECTS  # Per-EffectKind effect blocks
)

# Event names whose option list overflowed the cap (warn-once per name)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    # Name OHE
    out[_EVENT_NAME_TO_IDX[event.name]] = 1.0

    # State scalar
    out[len(_EVENT_NAME_TO_IDX)] = min(event.state, EVENT_STATE_CAP) / EVENT_STATE_CAP


def _encode_event_option_into(event_option: EventOption, slot: int, out: np.ndarray) -> None:
    # Slot-index OHE
    out[slot] = 1.0

    # Gated flag
    out[MAX_EVENT_OPTIONS] = float(event_option.gated_out)

    # Per-EffectKind effect blocks (what the option actually does)
    encode_effects_into(event_option.effects, MAX_EVENT_OPTIONS + 1, out)


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
        for i, option in enumerate(event.options[:MAX_EVENT_OPTIONS]):
            _encode_event_option_into(option, i, np_options[b, i])
            np_pad[b, i] = True

    return (
        torch.from_numpy(np_meta).to(device),
        torch.from_numpy(np_options).to(device),
        torch.from_numpy(np_pad).to(device),
    )
