import numpy as np
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.layout import make_channel_allocator


_CHANNEL_LAYOUT_PRICE, _alloc = make_channel_allocator()
_OFF_SHOP_PRICE = _alloc("price")
_OFF_SHOP_AFFORDABLE = _alloc("affordable")
_OFF_SHOP_GOLD_GAP = _alloc("gold_gap")  # `abs(gold - price)`

ENCODING_DIM_PRICE = sum(w for _, w in _CHANNEL_LAYOUT_PRICE)

# Most nonzero price columns a row holds: one per channel, each a scalar
ENCODING_DIM_SPARSE_PRICE = len(_CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
_OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1


def encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[_OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[_OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[_OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_shop_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write the purge service's price features and whether it was used."""
    shop = game_state.shop
    if shop is None:
        return
    encode_price_into(shop.purge_cost, game_state.character.gold, out)
    out[_OFF_SHOP_PURGED] = float(shop.purged)
