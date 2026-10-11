import warnings
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Event
from slai import EventName
from slai import GameEnv
from slai import GameState
from slai import MonsterEncounter

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import EVENT_STATE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.types import BatchEffectRows
from src.rl.types import EffectRow
from src.rl.types import EffectSource
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator
from src.rl.utils import get_sqrt_norm


_EVENT_NAME_TO_IDX = {name: int(name) for name in EventName.members()}

# Dead Adventurer's elite is one of three: the engine's src/events/dead_adventurer.rs
_MONSTER_ENCOUNTER_TO_IDX = {
    MonsterEncounter.GremlinNob: 0,
    MonsterEncounter.ThreeSentries: 1,
    MonsterEncounter.LagavulinEvent: 2,
}

# Match and Keep! deals six card names twice, face down: the engine's
# src/events/match_and_keep.rs, which GameEnv doesn't expose
_MATCH_AND_KEEP_BOARD_SIZE = 12


class EventMetaChannel(StrEnum):
    NAME = auto()
    CONSUMED = auto()
    STAGE = auto()
    HEALTH_MAX_AT_OPEN = auto()  # What ReadAt.EventOpen fractions read

    # Dead Adventurer only
    FOUND_GOLD = auto()
    FOUND_NOTHING = auto()
    FOUND_RELIC = auto()
    MONSTER_ENCOUNTER = auto()

    # Match and Keep! only
    MATCH_ATTEMPTS = auto()
    CARD_MATCH_UNSEEN_COUNT = auto()


CHANNEL_LAYOUT_EVENT_META, _alloc = make_channel_allocator()
OFF_NAME = _alloc(EventMetaChannel.NAME, len(_EVENT_NAME_TO_IDX))
OFF_CONSUMED = _alloc(EventMetaChannel.CONSUMED)
OFF_STAGE = _alloc(EventMetaChannel.STAGE)
OFF_HEALTH_MAX_AT_OPEN = _alloc(EventMetaChannel.HEALTH_MAX_AT_OPEN)
OFF_FOUND_GOLD = _alloc(EventMetaChannel.FOUND_GOLD)
OFF_FOUND_NOTHING = _alloc(EventMetaChannel.FOUND_NOTHING)
OFF_FOUND_RELIC = _alloc(EventMetaChannel.FOUND_RELIC)
OFF_MONSTER_ENCOUNTER = _alloc(
    EventMetaChannel.MONSTER_ENCOUNTER, len(_MONSTER_ENCOUNTER_TO_IDX)
)
OFF_MATCH_ATTEMPTS = _alloc(EventMetaChannel.MATCH_ATTEMPTS)
OFF_CARD_MATCH_UNSEEN_COUNT = _alloc(EventMetaChannel.CARD_MATCH_UNSEEN_COUNT)

ENCODING_DIM_EVENT_META = sum(w for _, w in CHANNEL_LAYOUT_EVENT_META)
ENCODING_DIM_EVENT_OPTION = MAX_EVENT_OPTIONS  # Slot-index OHE; Effects are separate rows

_SLOT_OHE = np.eye(MAX_EVENT_OPTIONS, dtype=np.float32)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    out[OFF_NAME + _EVENT_NAME_TO_IDX[event.name]] = 1.0
    out[OFF_CONSUMED] = float(event.consumed)
    out[OFF_STAGE] = min(event.stage, EVENT_STATE_CAP) / EVENT_STATE_CAP
    
    # On the character row's max-health scale, so the two compare directly
    out[OFF_HEALTH_MAX_AT_OPEN] = get_sqrt_norm(
        event.health_max_at_open, CHARACTER_HEALTH_CAP
    )

    # Dead Adventurer only
    out[OFF_FOUND_GOLD] = float(event.found_gold)
    out[OFF_FOUND_NOTHING] = float(event.found_nothing)
    out[OFF_FOUND_RELIC] = float(event.found_relic)
    adventurer_elite = event.adventurer_elite
    if adventurer_elite is not None:
        idx_adventurer_elite = _MONSTER_ENCOUNTER_TO_IDX.get(adventurer_elite)
        if idx_adventurer_elite is None:
            raise ValueError(
                f"Unknown adventurer elite {adventurer_elite!r}; add it to "
                "_MONSTER_ENCOUNTER_TO_IDX"
            )
        out[OFF_MONSTER_ENCOUNTER + idx_adventurer_elite] = 1.0

    # Match and Keep! only
    out[OFF_MATCH_ATTEMPTS] = event.match_attempts / GameEnv.MATCH_AND_KEEP_ATTEMPTS
    out[OFF_CARD_MATCH_UNSEEN_COUNT] = (
        event.card_match_unseen_count / _MATCH_AND_KEEP_BOARD_SIZE
    )


def encode_batch_events(
    game_states: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, BatchEffectRows]:
    """Event meta + (B, W, dim) option rows + their per-option kinds + each state's
    option effect rows, keyed by option slot; W is the batch's most options (capped
    at MAX_EVENT_OPTIONS). The option row itself is just a slot OHE — everything it
    does lives in its effect rows."""
    batch_size = len(game_states)
    batch_event = [game_state.event for game_state in game_states]  # hoisted: reads clone
    width = max(
        (min(len(event.options), MAX_EVENT_OPTIONS) for event in batch_event if event is not None),
        default=0,
    )

    # Pre-allocate NumPy arrays
    np_meta = np.zeros((batch_size, ENCODING_DIM_EVENT_META), dtype=np.float32)
    np_options = np.zeros((batch_size, width, ENCODING_DIM_EVENT_OPTION), dtype=np.float32)
    np_idxs = np.full((batch_size, width), -1, dtype=np.int16)
    batch_effect_rows: BatchEffectRows = []

    for b, event in enumerate(batch_event):
        effect_rows: list[EffectRow] = []
        batch_effect_rows.append(effect_rows)
        if event is None:
            continue

        if len(event.options) > MAX_EVENT_OPTIONS and event.name not in _WARNED_EVENT_TRUNCATED:
            _WARNED_EVENT_TRUNCATED.add(event.name)
            warnings.warn(
                f"Event {event.name!r} has {len(event.options)} options, truncated to"
                f" encoder cap ({MAX_EVENT_OPTIONS})"
            )

        # Event metadata
        _encode_event_meta_into(event, np_meta[b])

        # Event Options
        options = event.options[:MAX_EVENT_OPTIONS]
        np_options[b, : len(options)] = _SLOT_OHE[: len(options)]

        # `SliceKind`
        np_idxs[b, : len(options)] = SliceKind.EVENT_OPTIONS
        for w, option_effects in enumerate(options):
            if option_effects:
                effect_rows += [
                    (w, EffectSource.EVENT_OPTION_SELECT, effect) for effect in option_effects
                ]

    return (
        torch.from_numpy(np_meta).to(device),
        torch.from_numpy(np_options).to(device),
        np_idxs,
        batch_effect_rows,
    )
