from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState

from src.rl.constants import SHOP_PRICE_CAP
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator


# Price feats, written at each shop item's window token (items themselves go
# through their own entity encoders)
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
    game_states: list[GameState],
    np_cards_idxs: np.ndarray,
    np_relics_idxs: np.ndarray,
    np_potions_idxs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Price rows for the shop items that have a token, packed per state in kind
    order (cards, relics, potions) — (B, S, ENCODING_DIM_PRICE) rows + their (B, S)
    kind row — plus the shop meta. The i-th item of a kind prices the i-th token
    of that kind (the `np_*_idxs` are the selling types' kind rows), so an item
    the encoder dropped gets no price either."""
    batch_size = len(game_states)
    batch_shop = [game_state.shop for game_state in game_states]  # hoisted: reads clone
    items: list[list[tuple[SliceKind, int]]] = []
    for b, shop in enumerate(batch_shop):
        rows: list[tuple[SliceKind, int]] = []
        if shop is not None:
            for kind, np_idxs, prices in (
                (SliceKind.CARD_SHOP, np_cards_idxs, shop.card_prices),
                (SliceKind.RELIC_SHOP, np_relics_idxs, shop.relic_prices),
                (SliceKind.POTION_SHOP, np_potions_idxs, shop.potion_prices),
            ):
                num_tokens = int(np.count_nonzero(np_idxs[b] == kind))
                rows += [(kind, price) for price in prices[:num_tokens]]
        items.append(rows)
    width = max((len(rows) for rows in items), default=0)
    np_prices = np.zeros((batch_size, width, ENCODING_DIM_PRICE), dtype=np.float32)
    np_kinds = np.full((batch_size, width), -1, dtype=np.int16)
    np_meta = np.zeros((batch_size, ENCODING_DIM_SHOP_META), dtype=np.float32)

    for b, (game_state, shop) in enumerate(zip(game_states, batch_shop)):
        if shop is None:
            continue
        gold = game_state.character.gold
        for k, (kind, price) in enumerate(items[b]):
            _encode_price_into(price, gold, np_prices[b, k])
            np_kinds[b, k] = kind

        # Purge service price (same features as the per-item prices) + purged flag
        _encode_price_into(shop.purge_cost, gold, np_meta[b])
        np_meta[b, OFF_SHOP_PURGED] = float(shop.purged)

    return torch.from_numpy(np_prices).to(device), np_kinds, torch.from_numpy(np_meta).to(device)
