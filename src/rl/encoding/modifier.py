import numpy as np
from slai import Modifier
from slai import ModifierKind
from slai import members

from src.rl.utils import get_sqrt_norm


MODIFIER_KIND_TO_IDX = {modifier_kind: i for i, modifier_kind in enumerate(members(ModifierKind))}
ENCODING_DIM_MODIFIERS = len(MODIFIER_KIND_TO_IDX)
STACKS_MAX = 99


def encode_modifiers_into(modifiers: list[Modifier], pos: int, out: np.ndarray) -> int:
    for modifier in modifiers:
        idx = MODIFIER_KIND_TO_IDX[modifier.kind]
        # Boolean modifiers (engine ceiling 1) are presence flags; rest sqrt-scaled
        if modifier.stacks_max <= 1:
            out[pos + idx] = float(modifier.stacks)
        else:
            out[pos + idx] = get_sqrt_norm(modifier.stacks, STACKS_MAX)

    return pos + len(MODIFIER_KIND_TO_IDX)
