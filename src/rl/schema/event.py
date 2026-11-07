"""Feature layout and static catalogs for event rows."""

from slai import EventName
from slai import MonsterEncounter

from src.rl.schema.layout import make_channel_allocator


EVENT_NAME_TO_IDX = {name: int(name) for name in EventName.members()}

# Dead Adventurer's elite is one of three: the engine's src/events/dead_adventurer.rs
MONSTER_ENCOUNTER_TO_IDX = {
    MonsterEncounter.GremlinNob: 0,
    MonsterEncounter.ThreeSentries: 1,
    MonsterEncounter.LagavulinEvent: 2,
}

# Match and Keep! deals six Card names twice, face down: the engine's
# src/events/match_and_keep.rs, which GameEnv doesn't expose
MATCH_AND_KEEP_BOARD_SIZE = 12


CHANNEL_LAYOUT_EVENT_META, _alloc = make_channel_allocator()
OFF_EVENT_NAME = _alloc("name", len(EVENT_NAME_TO_IDX))
OFF_EVENT_CONSUMED = _alloc("consumed")
OFF_EVENT_STAGE = _alloc("stage")
OFF_EVENT_HEALTH_MAX_AT_OPEN = _alloc("health_max_at_open")  # What ReadAt.EventOpen fractions read
# Dead Adventurer only
OFF_EVENT_FOUND_GOLD = _alloc("found_gold")
OFF_EVENT_FOUND_NOTHING = _alloc("found_nothing")
OFF_EVENT_FOUND_RELIC = _alloc("found_relic")
OFF_EVENT_MONSTER_ENCOUNTER = _alloc("monster_encounter", len(MONSTER_ENCOUNTER_TO_IDX))
# Match and Keep! only
OFF_EVENT_MATCH_ATTEMPTS = _alloc("match_attempts")
OFF_EVENT_CARD_MATCH_UNSEEN_COUNT = _alloc("card_match_unseen_count")

ENCODING_DIM_EVENT_META = sum(w for _, w in CHANNEL_LAYOUT_EVENT_META)
