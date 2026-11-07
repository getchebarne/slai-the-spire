"""Feature layout and static catalogs for reward rows."""

from src.rl.schema.layout import make_channel_allocator


CHANNEL_LAYOUT_REWARD_META, _alloc = make_channel_allocator()
OFF_REWARD_RELICS_EXCLUSIVE = _alloc(
    "relics_exclusive"
)  # Boss Rewards: take one Relic, lose the rest
OFF_REWARD_CARDS_FORCED = _alloc("cards_forced")  # The Library: one selection, no skip

ENCODING_DIM_REWARD_META = sum(w for _, w in CHANNEL_LAYOUT_REWARD_META)
