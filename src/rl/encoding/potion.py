import numpy as np
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


_IDX_OF_POTION_NAME = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_IDX_OF_POTION_RARITY = {rarity: int(rarity) for rarity in PotionRarity.members()}


_CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
_OFF_POTION_NAME = _alloc("name", len(_IDX_OF_POTION_NAME))
_OFF_POTION_RARITY = _alloc("rarity", len(_IDX_OF_POTION_RARITY))
_OFF_POTION_REQUIRES_TARGET = _alloc("requires_target")
_OFF_POTION_COMBAT_ONLY = _alloc("combat_only")
_OFF_POTION_DOUBLED = _alloc(
    "doubled"
)  # Sacred Bark: its Effect rows already carry the doubled amounts
_OFF_POTION_USABLE = _alloc("usable")  # Drinkable now, and no mask blocks it

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


def encode_potion_into(potion: Potion, usable: bool, np_out: np.ndarray) -> None:
    np_out[_OFF_POTION_NAME + _IDX_OF_POTION_NAME[potion.name]] = 1.0
    np_out[_OFF_POTION_RARITY + _IDX_OF_POTION_RARITY[potion.rarity]] = 1.0
    np_out[_OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    np_out[_OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    np_out[_OFF_POTION_DOUBLED] = float(potion.doubled)
    np_out[_OFF_POTION_USABLE] = float(usable)
