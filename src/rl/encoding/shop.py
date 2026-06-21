import numpy as np
import torch
from slai import Shop

from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import SHOP_PRICE_CAP


# Price feats, encoded alongside each item (the items go through their entity encoders).
ENCODING_DIM_PRICE = 1 + 1 + 1  # Price scalar  # Affordable  # abs(gold - price)


def _encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[0] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[1] = float(gold >= price)
    out[2] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_batch_shop(
    batch_shop: list[Shop | None],
    batch_gold: list[int],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Per-item shop prices (aligned with the shop card/relic/potion segments) + meta."""
    batch_size = len(batch_shop)

    # Pre-allocate NumPy arrays
    np_price_cards = np.zeros(
        (batch_size, MAX_SIZE_SHOP_CARDS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    np_price_relics = np.zeros((batch_size, MAX_SHOP_RELICS, ENCODING_DIM_PRICE), dtype=np.float32)
    np_price_potions = np.zeros(
        (batch_size, MAX_SHOP_POTIONS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    np_price_purge = np.zeros((batch_size, ENCODING_DIM_PRICE), dtype=np.float32)

    for b, (shop, gold) in enumerate(zip(batch_shop, batch_gold)):
        if shop is None:
            continue

        for i, price in enumerate(shop.card_prices):
            _encode_price_into(price, gold, np_price_cards[b, i])

        for i, price in enumerate(shop.relic_prices):
            _encode_price_into(price, gold, np_price_relics[b, i])

        for i, price in enumerate(shop.potion_prices):
            _encode_price_into(price, gold, np_price_potions[b, i])

        # Purge service price (same features as the per-item prices)
        _encode_price_into(shop.purge_cost, gold, np_price_purge[b])

    return (
        torch.from_numpy(np_price_cards).to(device),
        torch.from_numpy(np_price_relics).to(device),
        torch.from_numpy(np_price_potions).to(device),
        torch.from_numpy(np_price_purge).to(device),
    )
