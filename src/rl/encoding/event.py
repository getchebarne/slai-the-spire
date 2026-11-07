import warnings

import numpy as np
from slai import EventName
from slai import EventOption
from slai import GameEnv
from slai import GameState
from slai import MonsterEncounter

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import EVENT_STAGE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.types import make_channel_allocator


_EVENT_NAME_TO_IDX = {name: int(name) for name in EventName.members()}

# Dead Adventurer's elite is one of three: the engine's src/events/dead_adventurer.rs
_MONSTER_ENCOUNTER_TO_IDX = {
    MonsterEncounter.GremlinNob: 0,
    MonsterEncounter.ThreeSentries: 1,
    MonsterEncounter.LagavulinEvent: 2,
}

# Match and Keep! deals six Card names twice, face down: the engine's
# src/events/match_and_keep.rs, which GameEnv doesn't expose
_MATCH_AND_KEEP_BOARD_SIZE = 12


CHANNEL_LAYOUT_EVENT_META, _alloc = make_channel_allocator()
OFF_EVENT_NAME = _alloc("name", len(_EVENT_NAME_TO_IDX))
OFF_EVENT_CONSUMED = _alloc("consumed")
OFF_EVENT_STAGE = _alloc("stage")
OFF_EVENT_HEALTH_MAX_AT_OPEN = _alloc("health_max_at_open")  # What ReadAt.EventOpen fractions read
# Dead Adventurer only
OFF_EVENT_FOUND_GOLD = _alloc("found_gold")
OFF_EVENT_FOUND_NOTHING = _alloc("found_nothing")
OFF_EVENT_FOUND_RELIC = _alloc("found_relic")
OFF_EVENT_MONSTER_ENCOUNTER = _alloc("monster_encounter", len(_MONSTER_ENCOUNTER_TO_IDX))
# Match and Keep! only
OFF_EVENT_MATCH_ATTEMPTS = _alloc("match_attempts")
OFF_EVENT_CARD_MATCH_UNSEEN_COUNT = _alloc("card_match_unseen_count")

ENCODING_DIM_EVENT_META = sum(w for _, w in CHANNEL_LAYOUT_EVENT_META)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


def encode_event_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write event features, retaining the option-cap warning used by token encoding."""
    event = game_state.event
    if event is None:
        return
    if len(event.options) > MAX_EVENT_OPTIONS and event.name not in _WARNED_EVENT_TRUNCATED:
        _WARNED_EVENT_TRUNCATED.add(event.name)
        warnings.warn(
            f"Event {event.name!r} has {len(event.options)} options, truncated to"
            f" encoder cap ({MAX_EVENT_OPTIONS})"
        )

    out[OFF_EVENT_NAME + _EVENT_NAME_TO_IDX[event.name]] = 1.0
    out[OFF_EVENT_CONSUMED] = float(event.consumed)
    out[OFF_EVENT_STAGE] = min(event.stage, EVENT_STAGE_CAP) / EVENT_STAGE_CAP

    # On the Character row's max-health scale, so the two compare directly
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


def get_event_option_entities(game_state: GameState) -> list[EventOption]:
    """Return the options behind the option tokens, in slot order, cut to MAX_EVENT_OPTIONS.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[EventOption]
        The Event's first MAX_EVENT_OPTIONS options; empty outside Events.
    """
    event = game_state.event
    return event.options[:MAX_EVENT_OPTIONS] if event is not None else []
