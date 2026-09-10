from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Effect
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.types import EffectSource
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator


# Pack order within the relic block: decision pools first, the owned relics last
SLICE_KINDS_RELIC = (
    SliceKind.RELIC_REWARD,
    SliceKind.RELIC_SHOP,
    SliceKind.RELIC_EVENT_ROLL,
    SliceKind.RELIC_OWNED,
)


_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
# Widest firing threshold in the catalog (Nunchaku/Ink Bottle 10); Girya (3) and
# Matryoshka (2) count without firing and sit far below
_COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())


class RelicChannel(StrEnum):
    NAME = auto()
    TIER = auto()
    USED_UP = auto()
    COUNTER = auto()
    COUNTER_PROGRESS = auto()  # counter / counter_reset for firing counters, else 0


CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
OFF_RELIC_NAME = _alloc(RelicChannel.NAME, len(_RELIC_NAME_TO_IDX))
OFF_RELIC_TIER = _alloc(RelicChannel.TIER, len(_RELIC_TIER_TO_IDX))
OFF_RELIC_USED_UP = _alloc(RelicChannel.USED_UP)
OFF_RELIC_COUNTER = _alloc(RelicChannel.COUNTER)
OFF_RELIC_COUNTER_PROGRESS = _alloc(RelicChannel.COUNTER_PROGRESS)

ENCODING_DIM_RELIC = sum(w for _, w in CHANNEL_LAYOUT_RELIC)

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


def encode_relic_into_w_cache(relic: Relic, out: np.ndarray) -> None:
    """Write the full relic encoding into `out`, cached by (name, used_up, clamped counter)."""
    cache_key = (relic.name, relic.used_up, min(relic.counter, _COUNTER_MAX))
    row = _RELIC_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_relic_entities(kind: SliceKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap: a relic past it has no token
    match kind:
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
            raise ValueError(f"not a relic kind: {kind}")


def iter_packed_relics(game_state: GameState) -> Iterator[tuple[int, SliceKind, Relic]]:
    """(w, kind, relic) for every relic token: the kinds back to back in
    SLICE_KINDS_RELIC order, each collection in engine order."""
    w = 0
    for kind in SLICE_KINDS_RELIC:
        for relic in _get_relic_entities(kind, game_state):
            yield w, kind, relic
            w += 1


def encode_batch_relics(
    game_states: list[GameState], device: torch.device
) -> tuple[torch.Tensor, np.ndarray, list[list[tuple[int, EffectSource, Effect]]]]:
    """(B, W, ENCODING_DIM_RELIC) rows + the (B, W) kind of each token + each state's
    (offset, source, effect) rows (every trigger's, per relic), keyed by the same w
    as the token they belong to; W is the batch's longest packed relic sequence."""
    batch_tokens = [list(iter_packed_relics(game_state)) for game_state in game_states]
    width = max((len(tokens) for tokens in batch_tokens), default=0)
    np_out = np.zeros((len(game_states), width, ENCODING_DIM_RELIC), dtype=np.float32)
    np_idxs = np.full((len(game_states), width), -1, dtype=np.int16)
    batch_effect_rows: list[list[tuple[int, EffectSource, Effect]]] = []

    for b, tokens in enumerate(batch_tokens):
        effect_rows: list[tuple[int, EffectSource, Effect]] = []
        for w, kind, relic in tokens:
            encode_relic_into_w_cache(relic, np_out[b, w])
            np_idxs[b, w] = kind
            effect_rows += [
                (w, source, effect)
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

    return torch.from_numpy(np_out).to(device), np_idxs, batch_effect_rows
