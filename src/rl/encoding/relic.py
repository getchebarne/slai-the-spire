from enum import StrEnum
from enum import auto

import numpy as np
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.types import TokenKind
from src.rl.types import make_channel_allocator


_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
_COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())


class RelicChannel(StrEnum):
    NAME = auto()
    TIER = auto()
    USED_UP = auto()
    COUNTER = auto()
    COUNTER_PROGRESS = auto()  # counter / counter_reset for firing counters, else 0
    COUNTER_REMAINING = auto()  # Triggers left until it fires, bucket 0 the next one


CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
OFF_RELIC_NAME = _alloc(RelicChannel.NAME, len(_RELIC_NAME_TO_IDX))
OFF_RELIC_TIER = _alloc(RelicChannel.TIER, len(_RELIC_TIER_TO_IDX))
OFF_RELIC_USED_UP = _alloc(RelicChannel.USED_UP)
OFF_RELIC_COUNTER = _alloc(RelicChannel.COUNTER)
OFF_RELIC_COUNTER_PROGRESS = _alloc(RelicChannel.COUNTER_PROGRESS)
OFF_RELIC_COUNTER_REMAINING = _alloc(RelicChannel.COUNTER_REMAINING, _COUNTER_MAX)

ENCODING_DIM_RELIC = sum(w for _, w in CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(CHANNEL_LAYOUT_RELIC)

# Per-relic encoding cache keyed by (name, used_up, clamped counter) — the only mutable fields
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, out: np.ndarray) -> None:
    out[OFF_RELIC_NAME + _RELIC_NAME_TO_IDX[relic.name]] = 1.0
    out[OFF_RELIC_TIER + _RELIC_TIER_TO_IDX[relic.tier]] = 1.0
    out[OFF_RELIC_USED_UP] = float(relic.used_up)
    out[OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        out[OFF_RELIC_COUNTER_PROGRESS] = max(relic.counter, 0) / counter_reset

        # None for a spent one-shot (-1) or past Stone Calendar's single firing
        if 0 <= relic.counter < counter_reset:
            out[OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0


def encode_relic_into_w_cache(relic: Relic, out: np.ndarray) -> None:
    """Write the full relic encoding into `out`, cached by (name, used_up, clamped counter)."""
    cache_key = (relic.name, relic.used_up, min(relic.counter, _COUNTER_MAX))
    row = _RELIC_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, row)
        row.flags.writeable = False
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = row
    out[:] = row


def get_relic_entities(token_kind: TokenKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap: a relic past it has no token
    match token_kind:
        case TokenKind.RELIC_OWNED:
            return game_state.relics[:MAX_RELICS]
        case TokenKind.RELIC_REWARD:
            reward = game_state.reward
            return reward.relics[:MAX_RELIC_REWARDS] if reward is not None else []
        case TokenKind.RELIC_SHOP:
            shop = game_state.shop
            return shop.relics[:MAX_SHOP_RELICS] if shop is not None else []
        case TokenKind.RELIC_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:
                return event.relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL]
            return []
        case _:
            raise ValueError(f"not a relic token kind: {token_kind}")
