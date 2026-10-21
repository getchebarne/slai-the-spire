from collections.abc import Sequence
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.sparse import owned_sparse
from src.rl.types import BatchPriceRows
from src.rl.types import X_OwnedSparse
from src.rl.types import make_channel_allocator


class PriceChannel(StrEnum):
    PRICE = auto()
    AFFORDABLE = auto()
    GOLD_GAP = auto()  # `abs(gold - price)`


CHANNEL_LAYOUT_PRICE, _alloc = make_channel_allocator()
OFF_SHOP_PRICE = _alloc(PriceChannel.PRICE)
OFF_SHOP_AFFORDABLE = _alloc(PriceChannel.AFFORDABLE)
OFF_SHOP_GOLD_GAP = _alloc(PriceChannel.GOLD_GAP)

ENCODING_DIM_PRICE = sum(w for _, w in CHANNEL_LAYOUT_PRICE)

# Most nonzero columns a price row holds: one per channel, each a scalar
ENCODING_SPARSE_PRICE = len(CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1


def _encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_batch_prices(
    game_states: list[GameState],
    sources: Sequence[BatchPriceRows],
    device: torch.device,
) -> X_OwnedSparse:
    """Every state's shop-token price rows as one (N, ENCODING_DIM_PRICE) block, held sparse,
    every state's after the previous state's, with their owner keys; and each state's (B,) row
    count."""
    batch_size = len(game_states)
    batch_rows = [
        [row for batch_price_rows in sources for row in batch_price_rows[b]]
        for b in range(batch_size)
    ]
    np_count = np.array([len(rows) for rows in batch_rows], dtype=np.int64)  # (B,)
    np_price = np.zeros((np_count.sum(), ENCODING_DIM_PRICE), dtype=np.float32)
    np_owner_slice_kind = np.empty(np_count.sum(), dtype=np.int64)
    np_owner_rank = np.empty(np_count.sum(), dtype=np.int64)

    row = 0
    for game_state, rows in zip(game_states, batch_rows):
        if not rows:
            continue

        gold = game_state.character.gold
        for slice_kind, rank, price in rows:
            _encode_price_into(price, gold, np_price[row])
            np_owner_slice_kind[row] = slice_kind
            np_owner_rank[row] = rank
            row += 1

    return owned_sparse(
        np_price, np_owner_slice_kind, np_owner_rank, np_count, ENCODING_SPARSE_PRICE, device
    )


def encode_batch_shop_meta(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    """The purge service's price row + the purged flag; 0 outside shops."""
    np_meta = np.zeros((len(game_states), ENCODING_DIM_SHOP_META), dtype=np.float32)
    for b, game_state in enumerate(game_states):
        shop = game_state.shop
        if shop is None:
            continue

        # Purge service price (same features as the per-item prices) + purged flag
        _encode_price_into(shop.purge_cost, game_state.character.gold, np_meta[b])
        np_meta[b, OFF_SHOP_PURGED] = float(shop.purged)

    return torch.from_numpy(np_meta).to(device)
