import numpy as np
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


_POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


_CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
_OFF_POTION_NAME = _alloc("name", len(_POTION_NAME_TO_IDX))
_OFF_POTION_RARITY = _alloc("rarity", len(_POTION_RARITY_TO_IDX))
_OFF_POTION_REQUIRES_TARGET = _alloc("requires_target")
_OFF_POTION_COMBAT_ONLY = _alloc("combat_only")
_OFF_POTION_DOUBLED = _alloc(
    "doubled"
)  # Sacred Bark: its Effect rows already carry the doubled amounts

ENCODING_DIM_POTION = sum(w for _, w in _CHANNEL_LAYOUT_POTION)

# Most nonzero columns a Potion row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_POTION = len(_CHANNEL_LAYOUT_POTION)

# Each tagged Potion's Tags, by name: Sacred Bark's variant keeps them
TAGS_OF_POTION_NAME: dict[PotionName, tuple[Tag, ...]] = {
    PotionName.Cunning: (Tag.NINJA,),
    PotionName.Duplication: (Tag.SETUP,),
    PotionName.GamblersBrew: (Tag.DISCARD,),
    PotionName.LiquidMemories: (Tag.SETUP,),
    PotionName.Poison: (Tag.POISON,),
}

# Per-potion encoding cache keyed by (name, doubled): every other field is fixed by name
_POTION_ROW_CACHE: dict[tuple[PotionName, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[_OFF_POTION_NAME + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[_OFF_POTION_RARITY + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[_OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[_OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    out[_OFF_POTION_DOUBLED] = float(potion.doubled)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full Potion encoding into `out`, cached by (name, doubled).

    Parameters
    ----------
    potion : Potion
        The Potion to encode.
    out : np.ndarray
        (ENCODING_DIM_POTION,) float32. The Potion's row, overwritten.
    """
    cache_key = (potion.name, potion.doubled)  # Sacred Bark's variant keeps the name
    row = _POTION_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, row)
        row.flags.writeable = False  # Guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[cache_key] = row
    out[:] = row
