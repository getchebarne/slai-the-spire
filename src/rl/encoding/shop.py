import numpy as np
import torch
from slai import Shop

from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import SHOP_PRICE_CAP

# Price feats, encoded alongside (not inside) each item's entity encoding — the
# shop items themselves are encoded by their entity-class encoders (card/relic/
# potion segments per src.rl.index)
ENCODING_DIM_PRICE = 1 + 1 + 1  # Price scalar  # Affordable  # abs(gold - price)
_DIM_SHOP_META = 1 + 1  # Purge cost scalar  # Purge affordable


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
    x_card_prices = np.zeros(
        (batch_size, MAX_SIZE_SHOP_CARDS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    x_relic_prices = np.zeros((batch_size, MAX_SHOP_RELICS, ENCODING_DIM_PRICE), dtype=np.float32)
    x_potion_prices = np.zeros(
        (batch_size, MAX_SHOP_POTIONS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    x_meta = np.zeros((batch_size, _DIM_SHOP_META), dtype=np.float32)

    for b, (shop, gold) in enumerate(zip(batch_shop, batch_gold)):
        if shop is None:
            continue

        for i, price in enumerate(shop.card_prices):
            _encode_price_into(price, gold, x_card_prices[b, i])

        for i, price in enumerate(shop.relic_prices):
            _encode_price_into(price, gold, x_relic_prices[b, i])

        for i, price in enumerate(shop.potion_prices):
            _encode_price_into(price, gold, x_potion_prices[b, i])

        # Metadata
        x_meta[b, 0] = min(shop.purge_cost, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
        x_meta[b, 1] = float(gold >= shop.purge_cost)

    return (
        torch.from_numpy(x_card_prices).to(device),
        torch.from_numpy(x_relic_prices).to(device),
        torch.from_numpy(x_potion_prices).to(device),
        torch.from_numpy(x_meta).to(device),
    )
