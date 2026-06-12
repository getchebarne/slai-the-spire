import numpy as np
import torch
from slai import Event
from slai import EventName
from slai import members
from slai import EventOption

from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into

_EVENT_NAME_TO_IDX = {event_name: i for i, event_name in enumerate(members(EventName))}
_ENCODING_DIM_EVENT_META = len(_EVENT_NAME_TO_IDX) + 1  # Name OHE  & state scalar
_ENCODING_DIM_EVENT_OPTION = (
    MAX_EVENT_OPTIONS  # Slot-index OHE
    + 1  # Gated out
    + ENCODING_DIM_EFFECTS  # Per-EffectKind effect blocks
)


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
    x_meta = np.zeros((batch_size, _ENCODING_DIM_EVENT_META), dtype=np.float32)
    x_options = np.zeros(
        (batch_size, MAX_EVENT_OPTIONS, _ENCODING_DIM_EVENT_OPTION), dtype=np.float32
    )
    x_pad = np.zeros((batch_size, MAX_EVENT_OPTIONS), dtype=bool)

    for b, event in enumerate(batch_event):
        if event is None:
            continue

        _encode_event_meta_into(event, x_meta[b])
        for i, option in enumerate(event.options[:MAX_EVENT_OPTIONS]):
            _encode_event_option_into(option, i, x_options[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_meta).to(device),
        torch.from_numpy(x_options).to(device),
        torch.from_numpy(x_pad).to(device),
    )
