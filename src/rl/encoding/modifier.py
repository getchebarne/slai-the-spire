import numpy as np
from slai import Modifier
from slai import ModifierKind

from src.rl.utils import get_sqrt_norm


_MODIFIER_KIND_TO_IDX = {modifier_kind: i for i, modifier_kind in enumerate(ModifierKind)}
_STACKS_MAX = 50  # sqrt-scaled cap; keeps resolution on small decision-relevant stacks


def get_encoding_dim_modifiers() -> int:
    return len(ModifierKind)


def encode_modifiers_into(modifiers: list[Modifier], pos: int, out: np.ndarray) -> int:
    for modifier in modifiers:
        idx = _MODIFIER_KIND_TO_IDX[modifier.kind]
        # Boolean modifiers (engine ceiling 1) are presence flags; rest sqrt-scaled
        if modifier.stacks_max <= 1:
            out[pos + idx] = float(modifier.stacks)
        else:
            out[pos + idx] = get_sqrt_norm(modifier.stacks, _STACKS_MAX)

    return pos + len(ModifierKind)
