import numpy as np
from slai import Action
from slai import ActionType
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag
from src.rl.utils import action_masked


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

# Per-potion encoding cache keyed by (name, doubled, usable): every other field is fixed by name
_POTION_ROW_CACHE: dict[tuple[PotionName, bool, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, usable: bool, np_out: np.ndarray) -> None:
    np_out[_OFF_POTION_NAME + _IDX_OF_POTION_NAME[potion.name]] = 1.0
    np_out[_OFF_POTION_RARITY + _IDX_OF_POTION_RARITY[potion.rarity]] = 1.0
    np_out[_OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    np_out[_OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    np_out[_OFF_POTION_DOUBLED] = float(potion.doubled)
    np_out[_OFF_POTION_USABLE] = float(usable)


def encode_potion_into_w_cache(game_state: GameState, potion: Potion, np_out: np.ndarray) -> None:
    """Write the full Potion encoding into `np_out`, cached by (name, doubled, usable).

    `usable` is the engine's `Potion.usable` with the masks folded in, so it is 1 exactly when the
    policy can drink the Potion.

    Parameters
    ----------
    game_state : GameState
        The state the Potion is in.
    potion : Potion
        The Potion to encode.
    np_out : np.ndarray
        (ENCODING_DIM_POTION,) float32. The Potion's row, overwritten.
    """
    # A drink carries its target as a second id; the masks read the Potion's alone
    usable = potion.usable and not action_masked(
        game_state, Action(ActionType.PotionUse, [potion.id])
    )

    # Sacred Bark's variant keeps the name
    cache_key = (potion.name, potion.doubled, usable)
    np_row = _POTION_ROW_CACHE.get(cache_key)
    if np_row is None:
        np_row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, usable, np_row)
        np_row.flags.writeable = False  # Guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[cache_key] = np_row
    np_out[:] = np_row
