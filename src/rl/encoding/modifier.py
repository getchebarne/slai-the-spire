import math

import numpy as np
from slai import Modifier
from slai import ModifierKind

from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INVINCIBLE_STACKS_MAX
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.types import make_channel_allocator


_MODIFIER_KIND_TO_IDX = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
_STACKS_MAX = 99

# Poison past the widest Monster HP kills on its next tick
_STACKS_MAX_OF_MODIFIER_KIND = {
    ModifierKind.Poison: HEALTH_MAX_ENTITY,
    ModifierKind.Invincible: INVINCIBLE_STACKS_MAX,  # The HP left to lose before its next turn
}
_STACKS_BUCKET_DIM = int(math.sqrt(max(_STACKS_MAX, *_STACKS_MAX_OF_MODIFIER_KIND.values()))) + 1


CHANNEL_LAYOUT_MODIFIER, _alloc = make_channel_allocator()
_OFF_MODIFIER_KIND = _alloc("kind", len(_MODIFIER_KIND_TO_IDX))
_OFF_MODIFIER_BUFF = _alloc("buff")
_OFF_MODIFIER_STACKS = _alloc("stacks")
_OFF_MODIFIER_STACKS_NEGATIVE = _alloc("stacks_negative")
_OFF_MODIFIER_STACKS_OHE = _alloc("stacks_ohe", _STACKS_BUCKET_DIM)

ENCODING_DIM_MODIFIER = sum(w for _, w in CHANNEL_LAYOUT_MODIFIER)

# Most nonzero columns a Modifier row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_MODIFIER = len(CHANNEL_LAYOUT_MODIFIER)


def encode_modifier_row_into(modifier: Modifier, out: np.ndarray) -> None:
    # Modifier kind
    out[_OFF_MODIFIER_KIND + _MODIFIER_KIND_TO_IDX[modifier.kind]] = 1.0

    # Buff or debuff
    out[_OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only sqrt buckets
    stacks_max = _STACKS_MAX_OF_MODIFIER_KIND.get(modifier.kind, _STACKS_MAX)
    out[_OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, stacks_max)
    out[_OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    out[_OFF_MODIFIER_STACKS_OHE + int(math.sqrt(min(abs(modifier.stacks), stacks_max)))] = 1.0
