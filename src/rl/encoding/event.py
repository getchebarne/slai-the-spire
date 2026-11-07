import warnings

import numpy as np
from slai import GameEnv
from slai import GameState

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import EVENT_STAGE_CAP
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.schema.event import EVENT_NAME_TO_IDX
from src.rl.schema.event import MATCH_AND_KEEP_BOARD_SIZE
from src.rl.schema.event import MONSTER_ENCOUNTER_TO_IDX
from src.rl.schema.event import OFF_EVENT_CARD_MATCH_UNSEEN_COUNT
from src.rl.schema.event import OFF_EVENT_CONSUMED
from src.rl.schema.event import OFF_EVENT_FOUND_GOLD
from src.rl.schema.event import OFF_EVENT_FOUND_NOTHING
from src.rl.schema.event import OFF_EVENT_FOUND_RELIC
from src.rl.schema.event import OFF_EVENT_HEALTH_MAX_AT_OPEN
from src.rl.schema.event import OFF_EVENT_MATCH_ATTEMPTS
from src.rl.schema.event import OFF_EVENT_MONSTER_ENCOUNTER
from src.rl.schema.event import OFF_EVENT_NAME
from src.rl.schema.event import OFF_EVENT_STAGE


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

    out[OFF_EVENT_NAME + EVENT_NAME_TO_IDX[event.name]] = 1.0
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
        idx_adventurer_elite = MONSTER_ENCOUNTER_TO_IDX.get(adventurer_elite)
        if idx_adventurer_elite is None:
            raise ValueError(
                f"Unknown adventurer elite {adventurer_elite!r}; add it to "
                "MONSTER_ENCOUNTER_TO_IDX"
            )
        out[OFF_EVENT_MONSTER_ENCOUNTER + idx_adventurer_elite] = 1.0

    # Match and Keep! only
    out[OFF_EVENT_MATCH_ATTEMPTS] = event.match_attempts / GameEnv.MATCH_AND_KEEP_ATTEMPTS
    out[OFF_EVENT_CARD_MATCH_UNSEEN_COUNT] = (
        event.card_match_unseen_count / MATCH_AND_KEEP_BOARD_SIZE
    )
