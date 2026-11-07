import numpy as np
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
from src.rl.schema.shop import OFF_SHOP_AFFORDABLE
from src.rl.schema.shop import OFF_SHOP_GOLD_GAP
from src.rl.schema.shop import OFF_SHOP_PRICE
from src.rl.schema.shop import OFF_SHOP_PURGED


def encode_price_into(price: int, gold: int, out: np.ndarray) -> None:
    out[OFF_SHOP_PRICE] = min(price, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    out[OFF_SHOP_AFFORDABLE] = float(gold >= price)
    out[OFF_SHOP_GOLD_GAP] = min(abs(gold - price), SHOP_PRICE_CAP) / SHOP_PRICE_CAP


def encode_shop_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write the purge service's price features and whether it was used."""
    shop = game_state.shop
    if shop is None:
        return
    encode_price_into(shop.purge_cost, game_state.character.gold, out)
    out[OFF_SHOP_PURGED] = float(shop.purged)
