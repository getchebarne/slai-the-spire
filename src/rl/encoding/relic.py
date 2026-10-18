from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.encoding.sparse import entity_sparse
from src.rl.types import SLICE_KINDS_RELIC
from src.rl.types import BatchEffectRows
from src.rl.types import BatchPriceRows
from src.rl.types import EffectRow
from src.rl.types import EffectSource
from src.rl.types import PriceRow
from src.rl.types import SliceKind
from src.rl.types import X_EntitySparse
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
ENCODING_SPARSE_RELIC = len(CHANNEL_LAYOUT_RELIC)

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


def _get_relic_entities(slice_kind: SliceKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap: a relic past it has no token
    match slice_kind:
        case SliceKind.RELIC_OWNED:
            return game_state.relics[:MAX_RELICS]
        case SliceKind.RELIC_REWARD:
            reward = game_state.reward
            return reward.relics[:MAX_RELIC_REWARDS] if reward is not None else []
        case SliceKind.RELIC_SHOP:
            shop = game_state.shop
            return shop.relics[:MAX_SHOP_RELICS] if shop is not None else []
        case SliceKind.RELIC_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:
                return event.relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL]
            return []
        case _:
            raise ValueError(f"not a relic slice kind: {slice_kind}")


def iter_packed_relics(game_state: GameState) -> Iterator[tuple[SliceKind, int, Relic]]:
    """(slice kind, rank, relic) for every relic token: the slice kinds back to back in
    SLICE_KINDS_RELIC order, each collection in engine order."""
    for slice_kind in SLICE_KINDS_RELIC:
        for rank, relic in enumerate(_get_relic_entities(slice_kind, game_state)):
            yield slice_kind, rank, relic


def encode_batch_relics(
    game_states: list[GameState], device: torch.device
) -> tuple[X_EntitySparse, BatchEffectRows, BatchPriceRows]:
    """The relic block ((B, W, ENCODING_DIM_RELIC) rows held sparse + the (B, W) slice kind of
    each token) + each state's (slice kind, rank, source, effect) rows (every trigger's, per
    relic) and its shop tokens' (slice kind, rank, price) rows, keyed by the token they belong
    to; W is the batch's longest packed relic sequence."""
    batch_tokens = [list(iter_packed_relics(game_state)) for game_state in game_states]
    batch_relic_prices = []
    for game_state in game_states:
        shop = game_state.shop
        batch_relic_prices.append(shop.relic_prices if shop is not None else [])

    width = max((len(tokens) for tokens in batch_tokens), default=0)
    np_out = np.zeros((len(game_states), width, ENCODING_DIM_RELIC), dtype=np.float32)
    np_slice_kind = np.full((len(game_states), width), -1, dtype=np.int16)
    batch_effect_rows: BatchEffectRows = []
    batch_price_rows: BatchPriceRows = []

    for b, tokens in enumerate(batch_tokens):
        effect_rows: list[EffectRow] = []
        price_rows: list[PriceRow] = []

        # Shop tokens come in shop order, so they take the parallel price list in step
        relic_prices = iter(batch_relic_prices[b])
        for w, (slice_kind, rank, relic) in enumerate(tokens):
            encode_relic_into_w_cache(relic, np_out[b, w])

            # `SliceKind`
            np_slice_kind[b, w] = slice_kind

            # Price
            if slice_kind == SliceKind.RELIC_SHOP:
                price_rows.append((slice_kind, rank, next(relic_prices)))

            # Effects
            effect_rows += [
                (slice_kind, rank, source, effect)
                for effects, source in (
                    (relic.effects_combat_start, EffectSource.RELIC_COMBAT_START),
                    (relic.effects_turn_start, EffectSource.RELIC_TURN_START),
                    (relic.effects_turn_end, EffectSource.RELIC_TURN_END),
                    (relic.effects_pickup, EffectSource.RELIC_PICKUP),
                    (relic.effects_rest, EffectSource.RELIC_REST),
                    (relic.effects_counter, EffectSource.RELIC_COUNTER),
                    (relic.effects_combat_end, EffectSource.RELIC_COMBAT_END),
                )
                for effect in effects
            ]

        batch_effect_rows.append(effect_rows)
        batch_price_rows.append(price_rows)

    return (
        entity_sparse(np_out, np_slice_kind, ENCODING_SPARSE_RELIC, device),
        batch_effect_rows,
        batch_price_rows,
    )
