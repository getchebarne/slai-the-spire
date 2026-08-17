import warnings
from typing import get_args

import numpy as np
import torch
from slai import EVENT_KIND_ORDER
from slai import Event
from slai import EventKind
from slai import EventKindColosseum
from slai import EventKindCursedTome
from slai import EventKindDeadAdventurer
from slai import EventKindGoldenIdol
from slai import EventKindKnowingSkull
from slai import EventKindScrapOoze
from slai import EventKindWeMeetAgain

from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.utils import get_sqrt_norm


# Order = fill order = Core's global-offset order
SLICE_EVENTS = [Slice(SliceKind.EVENT_OPTIONS, MAX_EVENT_OPTIONS)]

# Event identity = position in the engine's declaration-ordered tuple (the kind
# classes carry no discriminant of their own). Guard against the tuple going
# stale on an engine bump — it happened once (25/42).
_EVENT_KIND_TO_IDX = {cls: i for i, cls in enumerate(EVENT_KIND_ORDER)}
assert set(_EVENT_KIND_TO_IDX) == set(get_args(EventKind)), (
    f"EVENT_KIND_ORDER out of sync with the EventKind union: "
    f"missing={sorted(c.__name__ for c in set(get_args(EventKind)) - set(_EVENT_KIND_TO_IDX))}"
)

# Stage-bearing payloads share one progress scalar (the old Event.state); each
# variant's counter has its own name.
_STAGE_GETTERS = {
    EventKindGoldenIdol: lambda k: k.stage,
    EventKindScrapOoze: lambda k: k.attempts,
    EventKindColosseum: lambda k: k.stage,
    EventKindCursedTome: lambda k: k.stage,
    EventKindDeadAdventurer: lambda k: k.searches,
}

_SKULL_COST_CAP = 30  # Knowing Skull asks start at 6 HP and escalate +1 per use
_GOLD_ASK_CAP = 150  # We Meet Again rolls 50..=min(gold, 150)

_OFF_CONSUMED = len(_EVENT_KIND_TO_IDX)
_OFF_STAGE = _OFF_CONSUMED + 1
_OFF_DA_FOUND = _OFF_STAGE + 1  # DeadAdventurer found_{gold,nothing,relic}
_OFF_SKULL_COSTS = _OFF_DA_FOUND + 3  # {potion,gold,card}_cost_hp
_OFF_GOLD_ASK = _OFF_SKULL_COSTS + 3  # has-flag + normalized amount
ENCODING_DIM_EVENT_META = _OFF_GOLD_ASK + 2

ENCODING_DIM_EVENT_OPTION = (
    MAX_EVENT_OPTIONS + ENCODING_DIM_EFFECTS  # Slot-index OHE  # Per-EffectKind effect blocks
)
# (The old gated flag is gone with EventOption: option legality is masks-only
# now, from env.get_legal_actions().)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    kind = event.kind
    out[_EVENT_KIND_TO_IDX[type(kind)]] = 1.0
    out[_OFF_CONSUMED] = float(event.consumed)

    get_stage = _STAGE_GETTERS.get(type(kind))
    if get_stage is not None:
        out[_OFF_STAGE] = min(get_stage(kind), EVENT_STATE_CAP) / EVENT_STATE_CAP

    if isinstance(kind, EventKindDeadAdventurer):
        out[_OFF_DA_FOUND] = float(kind.found_gold)
        out[_OFF_DA_FOUND + 1] = float(kind.found_nothing)
        out[_OFF_DA_FOUND + 2] = float(kind.found_relic)
    elif isinstance(kind, EventKindKnowingSkull):
        costs = (kind.potion_cost_hp, kind.gold_cost_hp, kind.card_cost_hp)
        for i, cost in enumerate(costs):
            out[_OFF_SKULL_COSTS + i] = min(cost, _SKULL_COST_CAP) / _SKULL_COST_CAP
    elif isinstance(kind, EventKindWeMeetAgain) and kind.gold_ask is not None:
        out[_OFF_GOLD_ASK] = 1.0
        out[_OFF_GOLD_ASK + 1] = get_sqrt_norm(kind.gold_ask, _GOLD_ASK_CAP)
    # (WeMeetAgain's rolled pick_card/pick_potion are full entity tokens: the
    # CARD_EVENT / POTION_EVENT slices in card.py / potion.py.)


def _encode_event_option_into(effects: list, slot: int, out: np.ndarray) -> None:
    # Slot-index OHE
    out[slot] = 1.0

    # Per-EffectKind effect blocks (what the option actually does)
    encode_effects_into(effects, MAX_EVENT_OPTIONS, out)


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

        kind_cls = type(event.kind)
        if len(event.options) > MAX_EVENT_OPTIONS and kind_cls not in _WARNED_EVENT_TRUNCATED:
            _WARNED_EVENT_TRUNCATED.add(kind_cls)
            warnings.warn(
                f"Event {kind_cls.__name__} has {len(event.options)} options, truncated to"
                f" encoder cap ({MAX_EVENT_OPTIONS})"
            )

        _encode_event_meta_into(event, np_meta[b])
        for i, option_effects in enumerate(event.options[:MAX_EVENT_OPTIONS]):
            _encode_event_option_into(option_effects, i, np_options[b, i])
            np_pad[b, i] = True

    return (
        torch.from_numpy(np_meta).to(device),
        torch.from_numpy(np_options).to(device),
        torch.from_numpy(np_pad).to(device),
    )
