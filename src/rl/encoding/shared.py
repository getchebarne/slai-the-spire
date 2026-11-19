"""The shared block, the run of columns that ends every token row (health/block, price, stasis
slot, Tags): its layout, the single writer of each range, and a price's part of a record key.
"""

from collections.abc import Iterable

import numpy as np

from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.encoding.shop import encode_price_into
from src.rl.types import Tag


# The stasis slot columns: a one-hot over the Monsters' roster posts (Monster.slot)
_ENCODING_DIM_STASIS_SLOT = MAX_MONSTERS
ENCODING_DIM_SPARSE_STASIS_SLOT = 1

# The Tag columns: a Multi-Hot over the Tags
_ENCODING_DIM_TAG = len(Tag)
ENCODING_DIM_SPARSE_TAG = _ENCODING_DIM_TAG


_CHANNEL_LAYOUT_SHARED, _alloc = make_channel_allocator()
OFF_SHARED_HEALTH_BLOCK = _alloc("health_block", ENCODING_DIM_HEALTH_BLOCK)
OFF_SHARED_PRICE = _alloc("price", ENCODING_DIM_PRICE)
OFF_SHARED_STASIS_SLOT = _alloc("stasis_slot", _ENCODING_DIM_STASIS_SLOT)
OFF_SHARED_TAG = _alloc("tag", _ENCODING_DIM_TAG)

ENCODING_DIM_SHARED = sum(w for _, w in _CHANNEL_LAYOUT_SHARED)


def encode_shared_health_block_into(health: int, block: int, np_shared: np.ndarray) -> None:
    """Write a combatant's health and block at the shared block's health/block columns.

    Parameters
    ----------
    health : int
        The combatant's health.
    block : int
        The combatant's block.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    encode_health_block_into(
        health,
        block,
        np_shared[OFF_SHARED_HEALTH_BLOCK : OFF_SHARED_HEALTH_BLOCK + ENCODING_DIM_HEALTH_BLOCK],
    )


def encode_shared_price_into(price: int | None, gold: int, np_shared: np.ndarray) -> None:
    """Write a Shop token's price at the shared block's price columns.

    Parameters
    ----------
    price : int | None
        Its price; None out of the Shop: nothing is written.
    gold : int
        The Character's gold, which its price is read against.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    if price is not None:
        encode_price_into(
            price, gold, np_shared[OFF_SHARED_PRICE : OFF_SHARED_PRICE + ENCODING_DIM_PRICE]
        )


def encode_shared_stasis_slot_into(slot: int, np_shared: np.ndarray) -> None:
    """Write a Monster's slot, one-hot, at the shared block's stasis slot columns.

    Parameters
    ----------
    slot : int
        The slot of the token's Monster, or of the Monster holding the token's Card in stasis.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_shared[OFF_SHARED_STASIS_SLOT + slot] = 1.0


def encode_shared_tag_into(tags: Iterable[Tag], np_shared: np.ndarray) -> None:
    """Write a Card's, Relic's or Potion's Tags, multi-hot, at the shared block's Tag columns.

    Parameters
    ----------
    tags : Iterable[Tag]
        Its Tags.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    for tag in tags:
        np_shared[OFF_SHARED_TAG + tag] = 1.0


def get_record_key_price(price: int | None, gold: int) -> int | None:
    """The gold a Card's, Relic's or Potion's row reads: only a price reads it.

    Parameters
    ----------
    price : int | None
        Its price; None out of the Shop.
    gold : int
        The Character's gold.

    Returns
    -------
    int | None
        The gold when it has a price, None otherwise.
    """
    return None if price is None else gold
