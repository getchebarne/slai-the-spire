from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Shop

from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.types import make_channel_allocator


# Price feats, encoded alongside each item (the items go through their entity encoders)
class PriceChannel(StrEnum):
    PRICE = auto()
    AFFORDABLE = auto()
    GOLD_GAP = auto()  # abs(gold - price)


CHANNEL_LAYOUT_PRICE, _alloc = make_channel_allocator()
OFF_SHOP_PRICE = _alloc(PriceChannel.PRICE)
OFF_SHOP_AFFORDABLE = _alloc(PriceChannel.AFFORDABLE)
OFF_SHOP_GOLD_GAP = _alloc(PriceChannel.GOLD_GAP)

ENCODING_DIM_PRICE = sum(w for _, w in CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1


def _encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_batch_shop(
    batch_shop: list[Shop | None],
    batch_gold: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    """Per-item shop prices (aligned with the shop card/relic/potion segments) + meta."""
    batch_size = len(batch_shop)
    np_price_cards = np.zeros(
        (batch_size, MAX_SIZE_SHOP_CARDS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    np_price_relics = np.zeros((batch_size, MAX_SHOP_RELICS, ENCODING_DIM_PRICE), dtype=np.float32)
    np_price_potions = np.zeros(
        (batch_size, MAX_SHOP_POTIONS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    np_meta = np.zeros((batch_size, ENCODING_DIM_SHOP_META), dtype=np.float32)

    for b, (shop, gold) in enumerate(zip(batch_shop, batch_gold)):
        if shop is None:
            continue

        for i, price in enumerate(shop.card_prices):
            _encode_price_into(price, gold, np_price_cards[b, i])

        for i, price in enumerate(shop.relic_prices):
            _encode_price_into(price, gold, np_price_relics[b, i])

        for i, price in enumerate(shop.potion_prices):
            _encode_price_into(price, gold, np_price_potions[b, i])

        # Purge service price (same features as the per-item prices) + purged flag
        _encode_price_into(shop.purge_cost, gold, np_meta[b])
        np_meta[b, OFF_SHOP_PURGED] = float(shop.purged)

    return (
        torch.from_numpy(np_price_cards).to(device),
        torch.from_numpy(np_price_relics).to(device),
        torch.from_numpy(np_price_potions).to(device),
        torch.from_numpy(np_meta).to(device),
    )
