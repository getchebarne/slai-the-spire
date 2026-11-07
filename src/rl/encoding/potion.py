import numpy as np
from slai import Potion
from slai import PotionName

from src.rl.schema.potion import ENCODING_DIM_POTION
from src.rl.schema.potion import OFF_POTION_COMBAT_ONLY
from src.rl.schema.potion import OFF_POTION_DOUBLED
from src.rl.schema.potion import OFF_POTION_NAME
from src.rl.schema.potion import OFF_POTION_RARITY
from src.rl.schema.potion import OFF_POTION_REQUIRES_TARGET
from src.rl.schema.potion import POTION_NAME_TO_IDX
from src.rl.schema.potion import POTION_RARITY_TO_IDX


# Per-potion encoding cache keyed by (name, doubled): every other field is fixed by name
_POTION_ROW_CACHE: dict[tuple[PotionName, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[OFF_POTION_NAME + POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[OFF_POTION_RARITY + POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    out[OFF_POTION_DOUBLED] = float(potion.doubled)


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
