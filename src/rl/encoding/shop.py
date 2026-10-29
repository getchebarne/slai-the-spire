from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
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

# Most nonzero price columns a row holds: one per channel, each a scalar
ENCODING_DIM_SPARSE_PRICE = len(CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1


def encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_batch_shop_meta(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    """Encode each state's Shop meta: the purge service's price row + the purged flag.

    Parameters
    ----------
    game_states : list[GameState]
        The states to encode, B of them.
    device : torch.device
        The device of the returned tensor.

    Returns
    -------
    torch.Tensor
        (B, ENCODING_DIM_SHOP_META) float32. Each state's Shop meta; 0 outside Shops.
    """
    np_meta = np.zeros((len(game_states), ENCODING_DIM_SHOP_META), dtype=np.float32)
    for b, game_state in enumerate(game_states):
        shop = game_state.shop
        if shop is None:
            continue

        # Purge service price (same features as the per-item prices) + purged flag
        encode_price_into(shop.purge_cost, game_state.character.gold, np_meta[b])
        np_meta[b, OFF_SHOP_PURGED] = float(shop.purged)

    return torch.from_numpy(np_meta).to(device)
