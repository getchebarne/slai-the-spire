import numpy as np
import torch
from slai import Shop

from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import encode_card_into
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import encode_relic_into


_DIM_SHOP_CARD = (
    ENCODING_DIM_CARD      # Card features
    + 1                    # Price scalar
    + 1                    # Affordable
)
_DIM_SHOP_RELIC = (
    ENCODING_DIM_RELIC     # Relic features
    + 1                    # Price scalar
    + 1                    # Affordable
)
_DIM_SHOP_POTION = (
    ENCODING_DIM_POTION    # Potion features
    + 1                    # Price scalar
    + 1                    # Affordable
)
_DIM_SHOP_META = (
    1                      # Purge cost scalar
    + 1                    # Purge affordable
)


def _encode_price_into(price: int, gold: int, pos: int, out: np.ndarray) -> None:
    # Scalars
    out[pos] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[pos + 1] = float(gold >= price)


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
]:
    batch_size = len(batch_shop)

    # Pre-allocate NumPy arrays
    x_cards = np.zeros((batch_size, MAX_SHOP_CARDS, _DIM_SHOP_CARD), dtype=np.float32)
    x_cards_pad = np.zeros((batch_size, MAX_SHOP_CARDS), dtype=bool)
    x_relics = np.zeros((batch_size, MAX_SHOP_RELICS, _DIM_SHOP_RELIC), dtype=np.float32)
    x_relics_pad = np.zeros((batch_size, MAX_SHOP_RELICS), dtype=bool)
    x_potions = np.zeros((batch_size, MAX_SHOP_POTIONS, _DIM_SHOP_POTION), dtype=np.float32)
    x_potions_pad = np.zeros((batch_size, MAX_SHOP_POTIONS), dtype=bool)
    x_meta = np.zeros((batch_size, _DIM_SHOP_META), dtype=np.float32)

    for b, (shop, gold) in enumerate(zip(batch_shop, batch_gold)):
        if shop is None:
            continue

        for i, (card, price) in enumerate(zip(shop.cards, shop.card_prices)):
            pos = encode_card_into(card, 0, x_cards[b, i])
            _encode_price_into(price, gold, pos, x_cards[b, i])
            x_cards_pad[b, i] = True

        for i, (relic, price) in enumerate(zip(shop.relics, shop.relic_prices)):
            pos = encode_relic_into(relic, 0, x_relics[b, i])
            _encode_price_into(price, gold, pos, x_relics[b, i])
            x_relics_pad[b, i] = True

        for i, (potion, price) in enumerate(zip(shop.potions, shop.potion_prices)):
            pos = encode_potion_into(potion, 0, x_potions[b, i])
            _encode_price_into(price, gold, pos, x_potions[b, i])
            x_potions_pad[b, i] = True

        # Metadata
        x_meta[b, 0] = min(shop.purge_cost, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
        x_meta[b, 1] = float(gold >= shop.purge_cost)

    return (
        torch.from_numpy(x_cards).to(device),
        torch.from_numpy(x_cards_pad).to(device),
        torch.from_numpy(x_relics).to(device),
        torch.from_numpy(x_relics_pad).to(device),
        torch.from_numpy(x_potions).to(device),
        torch.from_numpy(x_potions_pad).to(device),
        torch.from_numpy(x_meta).to(device),
    )
