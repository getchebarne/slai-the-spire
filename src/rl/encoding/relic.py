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

from src.rl.constants import MAX_EVENT_ROLL_RELICS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import clip_to_slice
from src.rl.types import make_channel_allocator


# Order = fill order = Core's global-offset order
SLICE_RELICS = [
    Slice(SliceKind.RELIC_OWNED, MAX_RELICS),
    Slice(SliceKind.RELIC_REWARD, MAX_RELIC_REWARDS),
    Slice(SliceKind.RELIC_SHOP, MAX_SHOP_RELICS),
    Slice(SliceKind.RELIC_EVENT, MAX_EVENT_ROLL_RELICS),
]
NUM_RELIC_TOKENS = sum(slice_.size for slice_ in SLICE_RELICS)


_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
_COUNTER_MAX = 10  # Nunchaku 10, Girya 3, Matryoshka 2, Incense Burner 6, Sundial 3


class RelicChannel(StrEnum):
    NAME = auto()
    TIER = auto()
    USED_UP = auto()
    COUNTER = auto()


CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
OFF_RELIC_NAME = _alloc(RelicChannel.NAME, len(_RELIC_NAME_TO_IDX))
OFF_RELIC_TIER = _alloc(RelicChannel.TIER, len(_RELIC_TIER_TO_IDX))
OFF_RELIC_USED_UP = _alloc(RelicChannel.USED_UP)
OFF_RELIC_COUNTER = _alloc(RelicChannel.COUNTER)

ENCODING_DIM_RELIC = sum(w for _, w in CHANNEL_LAYOUT_RELIC)

# Per-relic encoding cache keyed by (name, used_up, clamped counter) — the only mutable fields
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, out: np.ndarray) -> None:
    out[OFF_RELIC_NAME + _RELIC_NAME_TO_IDX[relic.name]] = 1.0
    out[OFF_RELIC_TIER + _RELIC_TIER_TO_IDX[relic.tier]] = 1.0
    out[OFF_RELIC_USED_UP] = float(relic.used_up)
    out[OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX


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
    match kind:
        case SliceKind.RELIC_OWNED:
            return game_state.relics
        case SliceKind.RELIC_REWARD:
            reward = game_state.reward
            return reward.relics if reward is not None else []
        case SliceKind.RELIC_SHOP:
            return game_state.shop.relics if game_state.shop is not None else []
        case SliceKind.RELIC_EVENT:
            event = game_state.event
            if event is not None and not event.consumed:
                return event.roll_relics
            return []
        case _:
            raise ValueError(f"not a relic slice: {kind}")


def encode_batch_relics(
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all relic slices into one (B, NUM_RELIC_TOKENS, ENCODING_DIM_RELIC) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_RELIC_TOKENS, ENCODING_DIM_RELIC), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_RELIC_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        offset = 0
        for slice_ in SLICE_RELICS:
            relics = clip_to_slice(_get_relic_entities(slice_.kind, game_state), slice_)
            for i, relic in enumerate(relics):
                encode_relic_into_w_cache(relic, np_out[b, offset + i])
                np_pad[b, offset + i] = True
            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, effects) for every relic token with combat-start effects."""
    offset = 0
    for slice_ in SLICE_RELICS:
        relics = clip_to_slice(_get_relic_entities(slice_.kind, game_state), slice_)
        for i, relic in enumerate(relics):
            effects = relic.effects_combat_start  # bound once: each read clones the Vec
            if effects:
                yield offset + i, effects
        offset += slice_.size
