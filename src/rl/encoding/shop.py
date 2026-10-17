from collections.abc import Sequence
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.pairs import owned_pairs
from src.rl.types import BatchPriceRows
from src.rl.types import EntityKind
from src.rl.types import X_Prices
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
ENCODING_PAIRS_PRICE = len(CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1


def _encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_batch_prices(
    game_states: list[GameState],
    sources: Sequence[tuple[EntityKind, BatchPriceRows]],
    device: torch.device,
) -> X_Prices:
    """Every state's shop-token price rows as one (B, P, ENCODING_DIM_PRICE) block, held as
    pairs, with their owner keys; P is the batch's most priced tokens."""
    batch_size = len(game_states)
    batch_rows = [
        [
            (entity_kind, offset, price)
            for entity_kind, batch_price_rows in sources
            for offset, price in batch_price_rows[b]
        ]
        for b in range(batch_size)
    ]
    width = max((len(rows) for rows in batch_rows), default=0)
    np_price = np.zeros((batch_size, width, ENCODING_DIM_PRICE), dtype=np.float32)
    np_entity_kind = np.full((batch_size, width), -1, dtype=np.int64)
    np_offset = np.zeros((batch_size, width), dtype=np.int64)

    for b, (game_state, rows) in enumerate(zip(game_states, batch_rows)):
        if not rows:
            continue

        gold = game_state.character.gold
        for idx, (entity_kind, offset, price) in enumerate(rows):
            _encode_price_into(price, gold, np_price[b, idx])
            np_entity_kind[b, idx] = entity_kind
            np_offset[b, idx] = offset

    return owned_pairs(X_Prices, np_price, np_entity_kind, np_offset, ENCODING_PAIRS_PRICE, device)


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
