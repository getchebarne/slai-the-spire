import warnings

import numpy as np
from slai import MATCH_AND_KEEP_ATTEMPTS
from slai import NUM_MATCH_AND_KEEP_CARDS
from slai import EventName
from slai import EventOption
from slai import GameState
from slai import MonsterEncounter

from src.rl.constants import EVENT_STAGE_CAP
from src.rl.constants import HEALTH_SCALE
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import get_linear_norm
from src.rl.utils import event_option_wasted


_IDX_OF_EVENT_NAME = {name: int(name) for name in EventName.members()}

# Dead Adventurer's elite is one of three: the engine's src/events/dead_adventurer.rs
_IDX_OF_MONSTER_ENCOUNTER = {
    MonsterEncounter.GremlinNob: 0,
    MonsterEncounter.ThreeSentries: 1,
    MonsterEncounter.LagavulinEvent: 2,
}


_CHANNEL_LAYOUT_EVENT_META, _alloc = make_channel_allocator()
_OFF_EVENT_NAME = _alloc("name", len(_IDX_OF_EVENT_NAME))
_OFF_EVENT_CONSUMED = _alloc("consumed")
_OFF_EVENT_STAGE = _alloc("stage")
_OFF_EVENT_HEALTH_MAX_AT_OPEN = _alloc("health_max_at_open")
# Dead Adventurer only
_OFF_EVENT_FOUND_GOLD = _alloc("found_gold")
_OFF_EVENT_FOUND_NOTHING = _alloc("found_nothing")
_OFF_EVENT_FOUND_RELIC = _alloc("found_relic")
_OFF_EVENT_MONSTER_ENCOUNTER = _alloc("monster_encounter", len(_IDX_OF_MONSTER_ENCOUNTER))
# Match and Keep! only
_OFF_EVENT_MATCH_ATTEMPTS = _alloc("match_attempts")
_OFF_EVENT_CARD_MATCH_UNSEEN_COUNT = _alloc("card_match_unseen_count")

ENCODING_DIM_EVENT_META = sum(w for _, w, _ in _CHANNEL_LAYOUT_EVENT_META)

# Event kinds whose option list overflowed the cap (warn-once per kind)
_WARNED_EVENT_TRUNCATED: set = set()


_CHANNEL_LAYOUT_EVENT_OPTION, _alloc = make_channel_allocator()
_OFF_EVENT_OPTION_AVAILABLE = _alloc("available")

ENCODING_DIM_EVENT_OPTION = sum(w for _, w, _ in _CHANNEL_LAYOUT_EVENT_OPTION)

# Most nonzero columns an Event option row holds
ENCODING_DIM_SPARSE_EVENT_OPTION = sum(nonzeros for _, _, nonzeros in _CHANNEL_LAYOUT_EVENT_OPTION)


def encode_event_option_into(
    game_state: GameState, option: EventOption, np_out: np.ndarray
) -> None:
    """Write whether the policy can select `option`: legal in the engine and not masked as wasted.

    Parameters
    ----------
    game_state : GameState
        The state, at the option's Event.
    option : EventOption
        The option to encode.
    np_out : np.ndarray
        (ENCODING_DIM_EVENT_OPTION,) float32. The option's row, written in place.
    """
    np_out[_OFF_EVENT_OPTION_AVAILABLE] = float(
        option.available and not event_option_wasted(game_state, option.id)
    )


def encode_event_meta_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write event features, retaining the option-cap warning used by token encoding.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_EVENT_META,) float32. The context row's Event range, written in place; it
        stays 0 outside Events.
    """
    event = game_state.event
    if event is None:
        return
    if (
        len(event.id_event_options) > MAX_EVENT_OPTIONS
        and event.name not in _WARNED_EVENT_TRUNCATED
    ):
        _WARNED_EVENT_TRUNCATED.add(event.name)
        warnings.warn(
            f"Event {event.name!r} has {len(event.id_event_options)} options, truncated to"
            f" encoder cap ({MAX_EVENT_OPTIONS})"
        )

    np_out[_OFF_EVENT_NAME + _IDX_OF_EVENT_NAME[event.name]] = 1.0
    np_out[_OFF_EVENT_CONSUMED] = float(event.consumed)
    np_out[_OFF_EVENT_STAGE] = min(event.stage, EVENT_STAGE_CAP) / EVENT_STAGE_CAP

    # On the Character row's max-health scale, so the two compare directly
    np_out[_OFF_EVENT_HEALTH_MAX_AT_OPEN] = get_linear_norm(event.health_max_at_open, HEALTH_SCALE)

    # Dead Adventurer only
    np_out[_OFF_EVENT_FOUND_GOLD] = float(event.found_gold)
    np_out[_OFF_EVENT_FOUND_NOTHING] = float(event.found_nothing)
    np_out[_OFF_EVENT_FOUND_RELIC] = float(event.found_relic)
    adventurer_elite = event.adventurer_elite
    if adventurer_elite is not None:
        idx_adventurer_elite = _IDX_OF_MONSTER_ENCOUNTER.get(adventurer_elite)
        if idx_adventurer_elite is None:
            raise ValueError(
                f"Unknown adventurer elite {adventurer_elite!r}; add it to "
                "_IDX_OF_MONSTER_ENCOUNTER"
            )
        np_out[_OFF_EVENT_MONSTER_ENCOUNTER + idx_adventurer_elite] = 1.0

    # Match and Keep! only
    np_out[_OFF_EVENT_MATCH_ATTEMPTS] = event.match_attempts / MATCH_AND_KEEP_ATTEMPTS
    np_out[_OFF_EVENT_CARD_MATCH_UNSEEN_COUNT] = (
        event.card_match_unseen_count / NUM_MATCH_AND_KEEP_CARDS
    )
