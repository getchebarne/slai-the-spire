import warnings
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Event
from slai import EventName
from slai import EventOption
from slai import GameEnv
from slai import GameState
from slai import MonsterEncounter

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import EVENT_STAGE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.encoding.sparse import token_sparse
from src.rl.types import BatchEffectRows
from src.rl.types import EffectRow
from src.rl.types import EffectSource
from src.rl.types import TokenKind
from src.rl.types import X_Tokens
from src.rl.types import make_channel_allocator


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
OFF_EVENT_NAME = _alloc(EventMetaChannel.NAME, len(_EVENT_NAME_TO_IDX))
OFF_EVENT_CONSUMED = _alloc(EventMetaChannel.CONSUMED)
OFF_EVENT_STAGE = _alloc(EventMetaChannel.STAGE)
OFF_EVENT_HEALTH_MAX_AT_OPEN = _alloc(EventMetaChannel.HEALTH_MAX_AT_OPEN)
OFF_EVENT_FOUND_GOLD = _alloc(EventMetaChannel.FOUND_GOLD)
OFF_EVENT_FOUND_NOTHING = _alloc(EventMetaChannel.FOUND_NOTHING)
OFF_EVENT_FOUND_RELIC = _alloc(EventMetaChannel.FOUND_RELIC)
OFF_EVENT_MONSTER_ENCOUNTER = _alloc(
    EventMetaChannel.MONSTER_ENCOUNTER, len(_MONSTER_ENCOUNTER_TO_IDX)
)
OFF_EVENT_MATCH_ATTEMPTS = _alloc(EventMetaChannel.MATCH_ATTEMPTS)
OFF_EVENT_CARD_MATCH_UNSEEN_COUNT = _alloc(EventMetaChannel.CARD_MATCH_UNSEEN_COUNT)

ENCODING_DIM_EVENT_META = sum(w for _, w in CHANNEL_LAYOUT_EVENT_META)
ENCODING_DIM_EVENT_OPTION = MAX_EVENT_OPTIONS  # Slot-index OHE; Effects are separate rows
ENCODING_SPARSE_EVENT_OPTION = 1  # The slot one-hot's one nonzero column

_SLOT_OHE = np.eye(MAX_EVENT_OPTIONS, dtype=np.float32)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def _encode_event_meta_into(event: Event, out: np.ndarray) -> None:
    out[OFF_EVENT_NAME + _EVENT_NAME_TO_IDX[event.name]] = 1.0
    out[OFF_EVENT_CONSUMED] = float(event.consumed)
    out[OFF_EVENT_STAGE] = min(event.stage, EVENT_STAGE_CAP) / EVENT_STAGE_CAP

    # On the character row's max-health scale, so the two compare directly
    out[OFF_EVENT_HEALTH_MAX_AT_OPEN] = get_sqrt_norm(
        event.health_max_at_open, CHARACTER_HEALTH_CAP
    )

    # Dead Adventurer only
    out[OFF_EVENT_FOUND_GOLD] = float(event.found_gold)
    out[OFF_EVENT_FOUND_NOTHING] = float(event.found_nothing)
    out[OFF_EVENT_FOUND_RELIC] = float(event.found_relic)
    adventurer_elite = event.adventurer_elite
    if adventurer_elite is not None:
        idx_adventurer_elite = _MONSTER_ENCOUNTER_TO_IDX.get(adventurer_elite)
        if idx_adventurer_elite is None:
            raise ValueError(
                f"Unknown adventurer elite {adventurer_elite!r}; add it to "
                "_MONSTER_ENCOUNTER_TO_IDX"
            )
        out[OFF_EVENT_MONSTER_ENCOUNTER + idx_adventurer_elite] = 1.0

    # Match and Keep! only
    out[OFF_EVENT_MATCH_ATTEMPTS] = event.match_attempts / GameEnv.MATCH_AND_KEEP_ATTEMPTS
    out[OFF_EVENT_CARD_MATCH_UNSEEN_COUNT] = (
        event.card_match_unseen_count / _MATCH_AND_KEEP_BOARD_SIZE
    )


def _get_event_option_entities(event: Event | None) -> list[EventOption]:
    """The options behind the option tokens, in slot order, cut to MAX_EVENT_OPTIONS"""
    return event.options[:MAX_EVENT_OPTIONS] if event is not None else []


def encode_batch_events(
    game_states: list[GameState], column_start: int, device: torch.device
) -> tuple[torch.Tensor, X_Tokens, BatchEffectRows]:
    """Event meta + the option tokens ((N, dim) rows held sparse, their columns from
    `column_start` in the shared space, every state's options after the previous state's, + their
    token kinds + each state's (B,) option count, capped at MAX_EVENT_OPTIONS) + each state's
    option effect rows, keyed by option slot. The option row itself is just a slot OHE — everything
    it does lives in its effect rows."""
    batch_size = len(game_states)
    batch_event = [game_state.event for game_state in game_states]  # hoisted: reads clone
    batch_options = [_get_event_option_entities(event) for event in batch_event]
    np_count = np.array([len(options) for options in batch_options], dtype=np.int64)  # (B,)

    # Pre-allocate NumPy arrays
    np_meta = np.zeros((batch_size, ENCODING_DIM_EVENT_META), dtype=np.float32)
    np_options = np.zeros((np_count.sum(), ENCODING_DIM_EVENT_OPTION), dtype=np.float32)
    np_token_kind = np.empty(np_count.sum(), dtype=np.int16)
    batch_effect_rows: BatchEffectRows = []

    row = 0
    for b, (event, options) in enumerate(zip(batch_event, batch_options)):
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
        np_options[row : row + len(options)] = _SLOT_OHE[: len(options)]

        # `TokenKind`
        np_token_kind[row : row + len(options)] = TokenKind.EVENT_OPTIONS
        row += len(options)
        for w, option in enumerate(options):
            effects = option.effects  # Bound once: each read clones the Vec
            if effects:
                effect_rows += [
                    (TokenKind.EVENT_OPTIONS, w, EffectSource.EVENT_OPTION_SELECT, effect)
                    for effect in effects
                ]

    return (
        torch.from_numpy(np_meta).to(device),
        token_sparse(
            np_options,
            np_token_kind,
            np_count,
            ENCODING_SPARSE_EVENT_OPTION,
            column_start,
            device,
        ),
        batch_effect_rows,
    )
