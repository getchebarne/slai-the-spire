from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.encoding.sparse import entity_sparse
from src.rl.types import SLICE_KINDS_POTION
from src.rl.types import BatchEffectRows
from src.rl.types import BatchPriceRows
from src.rl.types import EffectRow
from src.rl.types import EffectSource
from src.rl.types import PriceRow
from src.rl.types import SliceKind
from src.rl.types import X_EntitySparse
from src.rl.types import make_channel_allocator


_POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


class PotionChannel(StrEnum):
    NAME = auto()
    RARITY = auto()
    REQUIRES_TARGET = auto()
    COMBAT_ONLY = auto()
    DOUBLED = auto()  # Sacred Bark: its Effect rows already carry the doubled amounts


CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
OFF_POTION_NAME = _alloc(PotionChannel.NAME, len(_POTION_NAME_TO_IDX))
OFF_POTION_RARITY = _alloc(PotionChannel.RARITY, len(_POTION_RARITY_TO_IDX))
OFF_POTION_REQUIRES_TARGET = _alloc(PotionChannel.REQUIRES_TARGET)
OFF_POTION_COMBAT_ONLY = _alloc(PotionChannel.COMBAT_ONLY)
OFF_POTION_DOUBLED = _alloc(PotionChannel.DOUBLED)

ENCODING_DIM_POTION = sum(w for _, w in CHANNEL_LAYOUT_POTION)

# Most nonzero columns a potion row holds: one per channel, each a one-hot or a scalar
ENCODING_SPARSE_POTION = len(CHANNEL_LAYOUT_POTION)

# Per-potion encoding cache keyed by (name, doubled): every other field is fixed by name
_POTION_ROW_CACHE: dict[tuple[PotionName, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[OFF_POTION_NAME + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[OFF_POTION_RARITY + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    out[OFF_POTION_DOUBLED] = float(potion.doubled)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full potion encoding into `out`, cached by (name, doubled)."""
    cache_key = (potion.name, potion.doubled)  # Sacred Bark's variant keeps the name
    row = _POTION_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_potion_entities(slice_kind: SliceKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap (all engine-bounded, so never lossy)
    match slice_kind:
        case SliceKind.POTION_OWNED:
            return game_state.potions[:MAX_POTION_SLOTS]
        case SliceKind.POTION_REWARD:
            reward = game_state.reward
            return reward.potions[:MAX_POTION_REWARDS] if reward is not None else []
        case SliceKind.POTION_SHOP:
            shop = game_state.shop
            return shop.potions[:MAX_SHOP_POTIONS] if shop is not None else []
        case SliceKind.POTION_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:  # post-choice: the offer is dead
                return event.potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL]
            return []
        case _:
            raise ValueError(f"not a potion slice kind: {slice_kind}")


def iter_packed_potions(game_state: GameState) -> Iterator[tuple[SliceKind, int, Potion]]:
    """(slice kind, rank, potion) for every potion token: the slice kinds back to back in
    SLICE_KINDS_POTION order, each collection in engine order."""
    for slice_kind in SLICE_KINDS_POTION:
        for rank, potion in enumerate(_get_potion_entities(slice_kind, game_state)):
            yield slice_kind, rank, potion


def encode_batch_potions(
    game_states: list[GameState], device: torch.device
) -> tuple[X_EntitySparse, BatchEffectRows, BatchPriceRows]:
    """The potion block ((N, ENCODING_DIM_POTION) rows held sparse, every state's tokens after the
    previous state's, + the (N,) slice kind of each token + each state's (B,) token count) + each
    state's (slice kind, rank, source, effect) rows and its shop tokens' (slice kind, rank, price)
    rows, keyed by the token they belong to."""
    batch_tokens = [list(iter_packed_potions(game_state)) for game_state in game_states]
    batch_potion_prices = []
    for game_state in game_states:
        shop = game_state.shop
        batch_potion_prices.append(shop.potion_prices if shop is not None else [])

    np_count = np.array([len(tokens) for tokens in batch_tokens], dtype=np.int64)  # (B,)
    np_out = np.zeros((np_count.sum(), ENCODING_DIM_POTION), dtype=np.float32)  # (N, D)
    np_slice_kind = np.empty(np_count.sum(), dtype=np.int16)  # (N,)
    batch_effect_rows: BatchEffectRows = []
    batch_price_rows: BatchPriceRows = []

    row = 0
    for b, tokens in enumerate(batch_tokens):
        effect_rows: list[EffectRow] = []
        price_rows: list[PriceRow] = []

        # Shop tokens come in shop order, so they take the parallel price list in step
        potion_prices = iter(batch_potion_prices[b])
        for slice_kind, rank, potion in tokens:
            encode_potion_into_w_cache(potion, np_out[row])

            # `SliceKind`
            np_slice_kind[row] = slice_kind

            # Price
            if slice_kind == SliceKind.POTION_SHOP:
                price_rows.append((slice_kind, rank, next(potion_prices)))

            # Effects
            effects = potion.effects  # Bound once: each read clones the Vec
            if effects:
                effect_rows += [
                    (slice_kind, rank, EffectSource.POTION_USE, effect) for effect in effects
                ]
            row += 1

        batch_effect_rows.append(effect_rows)
        batch_price_rows.append(price_rows)

    return (
        entity_sparse(np_out, np_slice_kind, np_count, ENCODING_SPARSE_POTION, device),
        batch_effect_rows,
        batch_price_rows,
    )
