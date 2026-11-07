"""Feature layout and static catalogs for relic rows."""

from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.schema.layout import make_channel_allocator
from src.rl.types import Tag


RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())


CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
OFF_RELIC_NAME = _alloc("name", len(RELIC_NAME_TO_IDX))
OFF_RELIC_TIER = _alloc("tier", len(RELIC_TIER_TO_IDX))
OFF_RELIC_USED_UP = _alloc("used_up")
OFF_RELIC_COUNTER = _alloc("counter")
OFF_RELIC_COUNTER_PROGRESS = _alloc(
    "counter_progress"
)  # Counter / counter_reset for firing counters, else 0
OFF_RELIC_COUNTER_REMAINING = _alloc(
    "counter_remaining", COUNTER_MAX
)  # Triggers left until it fires, bucket 0 the next one
OFF_RELIC_LINKED = _alloc("linked")  # The Relic the Sapphire Key is linked to

ENCODING_DIM_RELIC = sum(w for _, w in CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a Relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(CHANNEL_LAYOUT_RELIC)

# Each tagged Relic's Tags, by name
TAGS_OF_RELIC_NAME: dict[RelicName, tuple[Tag, ...]] = {
    RelicName.GamblingChip: (Tag.DISCARD,),
    RelicName.HoveringKite: (Tag.DISCARD,),
    RelicName.InkBottle: (Tag.NINJA,),
    RelicName.Kunai: (Tag.NINJA,),
    RelicName.NinjaScroll: (Tag.NINJA,),
    RelicName.Nunchaku: (Tag.NINJA,),
    RelicName.OrnamentalFan: (Tag.NINJA,),
    RelicName.PenNib: (Tag.NINJA,),
    RelicName.RunicPyramid: (Tag.SETUP,),
    RelicName.Shuriken: (Tag.NINJA,),
    RelicName.SneckoSkull: (Tag.POISON,),
    RelicName.TheSpecimen: (Tag.POISON,),
    RelicName.Tingsha: (Tag.DISCARD,),
    RelicName.ToughBandages: (Tag.DISCARD,),
    RelicName.WristBlade: (Tag.NINJA,),
}
