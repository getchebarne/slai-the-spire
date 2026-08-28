import math

import numpy as np
from slai import Modifier
from slai import ModifierKind

from src.rl.utils import get_sqrt_norm


_MODIFIER_KIND_TO_IDX = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
_STACKS_MAX = 99
_OFF_MODIFIER_BUFF = len(_MODIFIER_KIND_TO_IDX)
_OFF_MODIFIER_STACKS = _OFF_MODIFIER_BUFF + 1
_OFF_MODIFIER_STACKS_NEGATIVE = _OFF_MODIFIER_STACKS + 1
_OFF_MODIFIER_STACKS_OHE = _OFF_MODIFIER_STACKS_NEGATIVE + 1
_STACKS_BUCKET_DIM = int(math.sqrt(_STACKS_MAX)) + 1

ENCODING_DIM_MODIFIER = _OFF_MODIFIER_STACKS_OHE + _STACKS_BUCKET_DIM


def encode_modifier_row_into(modifier: Modifier, out: np.ndarray) -> None:
    # Modifier kind
    out[_MODIFIER_KIND_TO_IDX[modifier.kind]] = 1.0

    # Buff or debuff
    out[_OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only sqrt buckets
    out[_OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, _STACKS_MAX)
    out[_OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    out[_OFF_MODIFIER_STACKS_OHE + int(math.sqrt(min(abs(modifier.stacks), _STACKS_MAX)))] = 1.0
