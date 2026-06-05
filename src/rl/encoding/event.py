import numpy as np
import torch
from slai import Event
from slai import EventName

from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS


_EVENT_NAME_TO_IDX = {event_name: i for i, event_name in enumerate(EventName)}
_ENCODING_DIM_EVENT_META = (
    len(EventName)             # Name OHE
    + 1                        # State scalar
)
_ENCODING_DIM_EVENT_OPTION = (
    1                          # Gated out
)


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    # Name OHE
    name_idx = _EVENT_NAME_TO_IDX.get(int(event.name))
    if name_idx is not None:
        out[name_idx] = 1.0

    # State scalar
    out[len(EventName)] = min(event.state, EVENT_STATE_CAP) / EVENT_STATE_CAP


def _encode_event_option_into(option, out: np.ndarray) -> None:
    # Gated flags
    out[0] = float(option.gated_out)


def encode_batch_events(
    batch_event: list[Event | None], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch_event)

    # Pre-allocate NumPy arrays
    x_meta = np.zeros((batch_size, _ENCODING_DIM_EVENT_META), dtype=np.float32)
    x_opts = np.zeros((batch_size, MAX_EVENT_OPTIONS, _ENCODING_DIM_EVENT_OPTION), dtype=np.float32)
    x_pad = np.zeros((batch_size, MAX_EVENT_OPTIONS), dtype=bool)

    for b, event in enumerate(batch_event):
        if event is None:
            continue

        _encode_event_meta_into(event, x_meta[b])
        for i, option in enumerate(event.options):
            _encode_event_option_into(option, x_opts[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_meta).to(device),
        torch.from_numpy(x_opts).to(device),
        torch.from_numpy(x_pad).to(device),
    )
