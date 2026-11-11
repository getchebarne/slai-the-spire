import numpy as np
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
_COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())


_CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
_OFF_RELIC_NAME = _alloc("name", len(_RELIC_NAME_TO_IDX))
_OFF_RELIC_TIER = _alloc("tier", len(_RELIC_TIER_TO_IDX))
_OFF_RELIC_USED_UP = _alloc("used_up")
_OFF_RELIC_COUNTER = _alloc("counter")
_OFF_RELIC_COUNTER_PROGRESS = _alloc(
    "counter_progress"
)  # Counter / counter_reset for firing counters, else 0
_OFF_RELIC_COUNTER_REMAINING = _alloc(
    "counter_remaining", _COUNTER_MAX
)  # Triggers left until it fires, bucket 0 the next one
_OFF_RELIC_LINKED = _alloc("linked")  # The Relic the Sapphire Key is linked to

ENCODING_DIM_RELIC = sum(w for _, w in _CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a Relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(_CHANNEL_LAYOUT_RELIC)

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

# Per-relic encoding cache keyed by (name, used_up, counter, linked) — mutable fields and the link
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, linked: bool, out: np.ndarray) -> None:
    out[_OFF_RELIC_NAME + _RELIC_NAME_TO_IDX[relic.name]] = 1.0
    out[_OFF_RELIC_TIER + _RELIC_TIER_TO_IDX[relic.tier]] = 1.0
    out[_OFF_RELIC_USED_UP] = float(relic.used_up)
    out[_OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        out[_OFF_RELIC_COUNTER_PROGRESS] = max(relic.counter, 0) / counter_reset

        # None for a spent one-shot (-1) or past Stone Calendar's single firing
        if 0 <= relic.counter < counter_reset:
            out[_OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0

    out[_OFF_RELIC_LINKED] = float(linked)


def encode_relic_into_w_cache(relic: Relic, relic_linked: Relic | None, out: np.ndarray) -> None:
    """Write the Relic encoding into `out`, cached by (name, used_up, counter, linked).

    Parameters
    ----------
    relic : Relic
        The Relic to encode.
    relic_linked : Relic | None
        The Relic the Sapphire Key is linked to, None without the key on offer.
    out : np.ndarray
        (ENCODING_DIM_RELIC,) float32. The Relic's row, overwritten.
    """
    linked = relic_linked is not None and relic.id == relic_linked.id
    cache_key = (relic.name, relic.used_up, relic.counter, linked)
    row = _RELIC_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, linked, row)
        row.flags.writeable = False
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = row
    out[:] = row
