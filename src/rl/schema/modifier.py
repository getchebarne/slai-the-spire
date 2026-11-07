"""Feature layout and static catalogs for modifier rows."""

import math

from slai import ModifierKind

from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INVINCIBLE_STACKS_MAX
from src.rl.schema.layout import make_channel_allocator
from src.rl.types import ModifierOwner


MODIFIER_KIND_TO_IDX = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
STACKS_MAX = 99

# Poison past the widest Monster HP kills on its next tick
STACKS_MAX_OF_MODIFIER_KIND = {
    ModifierKind.Poison: HEALTH_MAX_ENTITY,
    ModifierKind.Invincible: INVINCIBLE_STACKS_MAX,  # The HP left to lose before its next turn
}
_STACKS_BUCKET_DIM = int(math.sqrt(max(STACKS_MAX, *STACKS_MAX_OF_MODIFIER_KIND.values()))) + 1


CHANNEL_LAYOUT_MODIFIER, _alloc = make_channel_allocator()
OFF_MODIFIER_KIND = _alloc("kind", len(MODIFIER_KIND_TO_IDX))
OFF_MODIFIER_BUFF = _alloc("buff")
OFF_MODIFIER_STACKS = _alloc("stacks")
OFF_MODIFIER_STACKS_NEGATIVE = _alloc("stacks_negative")
OFF_MODIFIER_STACKS_OHE = _alloc("stacks_ohe", _STACKS_BUCKET_DIM)
OFF_MODIFIER_OWNER = _alloc("owner", len(ModifierOwner))

ENCODING_DIM_MODIFIER = sum(w for _, w in CHANNEL_LAYOUT_MODIFIER)

# Most nonzero columns a Modifier row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_MODIFIER = len(CHANNEL_LAYOUT_MODIFIER)
