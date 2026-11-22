import numpy as np
from slai import TIME_WARP_PLAYS
from slai import Modifier
from slai import ModifierKind

from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INVINCIBLE_STACKS_MAX
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import encode_piecewise_into
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.types import ModifierOwner


_IDX_OF_MODIFIER_KIND = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
_STACKS_MAX = 99

# Poison past the widest Monster HP kills on its next tick
_STACKS_MAX_OF_MODIFIER_KIND = {
    ModifierKind.Poison: HEALTH_MAX_ENTITY,
    ModifierKind.Invincible: INVINCIBLE_STACKS_MAX,  # The HP left to lose before its next turn
    ModifierKind.TimeWarp: TIME_WARP_PLAYS - 1,  # The next Card played ends the turn
}
_LINEAR_SQRT_THRESHOLD = 25
_STACKS_BUCKET_DIM = get_piecewise_dim(
    0, max(_STACKS_MAX, *_STACKS_MAX_OF_MODIFIER_KIND.values()), _LINEAR_SQRT_THRESHOLD
)


_CHANNEL_LAYOUT_MODIFIER, _alloc = make_channel_allocator()
_OFF_MODIFIER_KIND = _alloc("kind", len(_IDX_OF_MODIFIER_KIND))
_OFF_MODIFIER_BUFF = _alloc("buff")
_OFF_MODIFIER_STACKS = _alloc("stacks")
_OFF_MODIFIER_STACKS_NEGATIVE = _alloc("stacks_negative")
_OFF_MODIFIER_STACKS_OHE = _alloc("stacks_ohe", _STACKS_BUCKET_DIM, 2)
_OFF_MODIFIER_OWNER = _alloc("owner", len(ModifierOwner))

ENCODING_DIM_MODIFIER = sum(w for _, w, _ in _CHANNEL_LAYOUT_MODIFIER)

# Most nonzero columns a Modifier row holds: one per one-hot or scalar, two per two-hot
ENCODING_DIM_SPARSE_MODIFIER = sum(nonzeros for _, _, nonzeros in _CHANNEL_LAYOUT_MODIFIER)


def make_modifier_row(modifier: Modifier, modifier_owner: ModifierOwner) -> np.ndarray:
    """Make a Modifier row: its kind, whether it is a buff, its stacks and its owner.

    Parameters
    ----------
    modifier : Modifier
        The Modifier.
    modifier_owner : ModifierOwner
        The kind of combatant carrying it.

    Returns
    -------
    np.ndarray
        (ENCODING_DIM_MODIFIER,) float32. The row.
    """
    np_row = np.zeros(ENCODING_DIM_MODIFIER, dtype=np.float32)

    # Modifier kind
    np_row[_OFF_MODIFIER_KIND + _IDX_OF_MODIFIER_KIND[modifier.kind]] = 1.0

    # Buff or debuff
    np_row[_OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only linear-sqrt two-hot
    stacks_max = _STACKS_MAX_OF_MODIFIER_KIND.get(modifier.kind, _STACKS_MAX)
    np_row[_OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, stacks_max)
    np_row[_OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    encode_piecewise_into(
        abs(modifier.stacks),
        0,
        stacks_max,
        _LINEAR_SQRT_THRESHOLD,
        np_row[_OFF_MODIFIER_STACKS_OHE:][:_STACKS_BUCKET_DIM],
    )

    # The combatant carrying this modifier
    np_row[_OFF_MODIFIER_OWNER + modifier_owner] = 1.0

    return np_row
