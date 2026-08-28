import warnings
from collections.abc import Iterator

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


# Order = fill order = Core's global-offset order
SLICE_EVENTS = [Slice(SliceKind.EVENT_OPTIONS, MAX_EVENT_OPTIONS)]

# int() keying: declaration-order discriminant, stable across engine bumps
_EVENT_NAME_TO_IDX = {name: int(name) for name in EventName.members()}

# The old per-kind payloads are flattened onto Event: one `stage` progress
# scalar and DeadAdventurer's found_* flags. Knowing Skull's escalating asks
# and We Meet Again's gold ask no longer need meta mirrors — the engine bakes
# the live prices into the option effect lists (HealthDelta/GoldDelta amounts)
_OFF_CONSUMED = len(_EVENT_NAME_TO_IDX)
_OFF_STAGE = _OFF_CONSUMED + 1
_OFF_DA_FOUND = _OFF_STAGE + 1  # DeadAdventurer found_{gold,nothing,relic}
ENCODING_DIM_EVENT_META = _OFF_DA_FOUND + 3

ENCODING_DIM_EVENT_OPTION = MAX_EVENT_OPTIONS  # Slot-index OHE; effects are separate rows
# (The old gated flag is gone with EventOption: option legality is masks-only
# now, from env.get_legal_actions().)

_SLOT_OHE = np.eye(MAX_EVENT_OPTIONS, dtype=np.float32)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    out[_EVENT_NAME_TO_IDX[event.name]] = 1.0
    out[_OFF_CONSUMED] = float(event.consumed)
    out[_OFF_STAGE] = min(event.stage, EVENT_STATE_CAP) / EVENT_STATE_CAP
    out[_OFF_DA_FOUND] = float(event.found_gold)
    out[_OFF_DA_FOUND + 1] = float(event.found_nothing)
    out[_OFF_DA_FOUND + 2] = float(event.found_relic)
    # (Staked picks are full entity tokens: the CARD_EVENT / POTION_EVENT /
    # RELIC_EVENT slices in card.py / potion.py / relic.py.)


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
        # The option row is only its slot OHE now; what the option DOES rides
        # in its effect rows (effect.py)
        n_options = min(len(event.options), MAX_EVENT_OPTIONS)
        np_options[b, :n_options] = _SLOT_OHE[:n_options]
        np_pad[b, :n_options] = True

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
