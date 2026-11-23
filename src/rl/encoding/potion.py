import numpy as np
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.shared import encode_shared_price_into
from src.rl.encoding.shared import encode_shared_tag_into
from src.rl.encoding.shared import get_record_key_price
from src.rl.types import RecordKeyPotion
from src.rl.types import Tag


_IDX_OF_POTION_NAME = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_IDX_OF_POTION_RARITY = {rarity: int(rarity) for rarity in PotionRarity.members()}


_CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
_OFF_POTION_NAME = _alloc("name", len(_IDX_OF_POTION_NAME))
_OFF_POTION_RARITY = _alloc("rarity", len(_IDX_OF_POTION_RARITY))
_OFF_POTION_REQUIRES_TARGET = _alloc("requires_target")
_OFF_POTION_COMBAT_ONLY = _alloc("combat_only")
_OFF_POTION_DOUBLED = _alloc("doubled")
_OFF_POTION_USABLE = _alloc("usable")

ENCODING_DIM_POTION = sum(w for _, w in _CHANNEL_LAYOUT_POTION)

# Most nonzero columns a Potion row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_POTION = len(_CHANNEL_LAYOUT_POTION)

# Each tagged Potion's Tags, by name: Sacred Bark's variant keeps them
_TAGS_OF_POTION_NAME: dict[PotionName, tuple[Tag, ...]] = {
    PotionName.Cunning: (Tag.NINJA,),
    PotionName.GamblersBrew: (Tag.DISCARD,),
    PotionName.Poison: (Tag.POISON,),
}


def encode_potion_into(
    potion: Potion, usable: bool, gold: int, np_own: np.ndarray, np_shared: np.ndarray
) -> None:
    """Write a Potion's token: its own columns, then its price and its Tags.

    Every input besides the Potion is in the Potion's record key (`get_record_key_potion`), so a
    new input goes in the key too.

    Parameters
    ----------
    potion : Potion
        The Potion.
    usable : bool
        Whether the policy can drink it: the engine's `Potion.usable` with the masks folded in.
    gold : int
        The Character's gold, which its price is read against.
    np_own : np.ndarray
        (ENCODING_DIM_POTION,) float32. The Potion's own range of its token row, written in place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_own[_OFF_POTION_NAME + _IDX_OF_POTION_NAME[potion.name]] = 1.0
    np_own[_OFF_POTION_RARITY + _IDX_OF_POTION_RARITY[potion.rarity]] = 1.0
    np_own[_OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    np_own[_OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    np_own[_OFF_POTION_DOUBLED] = float(potion.doubled)
    np_own[_OFF_POTION_USABLE] = float(usable)

    # Shop price
    encode_shared_price_into(potion.price, gold, np_shared)

    # Tags. Shared w/ Cards and Relics
    encode_shared_tag_into(_TAGS_OF_POTION_NAME.get(potion.name, ()), np_shared)


def get_record_key_potion(potion: Potion, usable: bool, gold: int) -> RecordKeyPotion:
    """Make a Potion's record key: its identity_hash and everything else its row reads.

    Parameters
    ----------
    potion : Potion
        The Potion.
    usable : bool
        Whether the policy can drink it: the engine's `Potion.usable` with the masks folded in.
    gold : int
        The Character's gold, which its price is read against.

    Returns
    -------
    RecordKeyPotion
        Its identity_hash, whether the policy can drink it, and the gold when it has a price.
    """
    return (potion.identity_hash, usable, get_record_key_price(potion.price, gold))
