import numpy as np
from slai import Relic

from src.rl.schema.relic import COUNTER_MAX
from src.rl.schema.relic import ENCODING_DIM_RELIC
from src.rl.schema.relic import OFF_RELIC_COUNTER
from src.rl.schema.relic import OFF_RELIC_COUNTER_PROGRESS
from src.rl.schema.relic import OFF_RELIC_COUNTER_REMAINING
from src.rl.schema.relic import OFF_RELIC_LINKED
from src.rl.schema.relic import OFF_RELIC_NAME
from src.rl.schema.relic import OFF_RELIC_TIER
from src.rl.schema.relic import OFF_RELIC_USED_UP
from src.rl.schema.relic import RELIC_NAME_TO_IDX
from src.rl.schema.relic import RELIC_TIER_TO_IDX


# Per-relic encoding cache keyed by (name, used_up, counter, linked) — mutable fields and the link
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, linked: bool, out: np.ndarray) -> None:
    out[OFF_RELIC_NAME + RELIC_NAME_TO_IDX[relic.name]] = 1.0
    out[OFF_RELIC_TIER + RELIC_TIER_TO_IDX[relic.tier]] = 1.0
    out[OFF_RELIC_USED_UP] = float(relic.used_up)
    out[OFF_RELIC_COUNTER] = min(relic.counter, COUNTER_MAX) / COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        out[OFF_RELIC_COUNTER_PROGRESS] = max(relic.counter, 0) / counter_reset

        # None for a spent one-shot (-1) or past Stone Calendar's single firing
        if 0 <= relic.counter < counter_reset:
            out[OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0

    out[OFF_RELIC_LINKED] = float(linked)


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
