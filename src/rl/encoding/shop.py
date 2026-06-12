import numpy as np
import torch
from slai import Shop

from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import encode_card_row
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import encode_relic_into


# Price feats, encoded alongside (not inside) each item's entity encoding
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = len(batch_shop)

    # Pre-allocate NumPy arrays
    x_cards = np.zeros((batch_size, MAX_SHOP_CARDS, ENCODING_DIM_CARD), dtype=np.float32)
    x_cards_pad = np.zeros((batch_size, MAX_SHOP_CARDS), dtype=bool)
    x_card_prices = np.zeros((batch_size, MAX_SHOP_CARDS, ENCODING_DIM_PRICE), dtype=np.float32)
    x_relics = np.zeros((batch_size, MAX_SHOP_RELICS, ENCODING_DIM_RELIC), dtype=np.float32)
    x_relics_pad = np.zeros((batch_size, MAX_SHOP_RELICS), dtype=bool)
    x_relic_prices = np.zeros((batch_size, MAX_SHOP_RELICS, ENCODING_DIM_PRICE), dtype=np.float32)
    x_potions = np.zeros((batch_size, MAX_SHOP_POTIONS, ENCODING_DIM_POTION), dtype=np.float32)
    x_potions_pad = np.zeros((batch_size, MAX_SHOP_POTIONS), dtype=bool)
    x_potion_prices = np.zeros(
        (batch_size, MAX_SHOP_POTIONS, ENCODING_DIM_PRICE), dtype=np.float32
    )
    x_meta = np.zeros((batch_size, _DIM_SHOP_META), dtype=np.float32)

    for b, (shop, gold) in enumerate(zip(batch_shop, batch_gold)):
        if shop is None:
            continue

        # Shop cards are bought with gold, not played; energy_current is irrelevant -> 0
        for i, (card, price) in enumerate(zip(shop.cards, shop.card_prices)):
            encode_card_row(card, 0, x_cards[b, i])
            _encode_price_into(price, gold, x_card_prices[b, i])
            x_cards_pad[b, i] = True

        for i, (relic, price) in enumerate(zip(shop.relics, shop.relic_prices)):
            encode_relic_into(relic, 0, x_relics[b, i])
            _encode_price_into(price, gold, x_relic_prices[b, i])
            x_relics_pad[b, i] = True

        for i, (potion, price) in enumerate(zip(shop.potions, shop.potion_prices)):
            encode_potion_into(potion, 0, x_potions[b, i])
            _encode_price_into(price, gold, x_potion_prices[b, i])
            x_potions_pad[b, i] = True

        # Metadata
        x_meta[b, 0] = min(shop.purge_cost, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
        x_meta[b, 1] = float(gold >= shop.purge_cost)

    return (
        torch.from_numpy(x_cards).to(device),
        torch.from_numpy(x_cards_pad).to(device),
        torch.from_numpy(x_card_prices).to(device),
        torch.from_numpy(x_relics).to(device),
        torch.from_numpy(x_relics_pad).to(device),
        torch.from_numpy(x_relic_prices).to(device),
        torch.from_numpy(x_potions).to(device),
        torch.from_numpy(x_potions_pad).to(device),
        torch.from_numpy(x_potion_prices).to(device),
        torch.from_numpy(x_meta).to(device),
    )
