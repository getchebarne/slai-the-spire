"""Feature layout and static catalogs for health block rows."""

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.schema.layout import make_channel_allocator
from src.rl.schema.scalars import get_piecewise_dim


HEALTH_MIN = 1
BLOCK_MIN = 0
TOTAL_MIN = HEALTH_MIN + BLOCK_MIN
TOTAL_MAX = HEALTH_MAX_ENTITY + BLOCK_MAX_ENTITY
LINEAR_SQRT_THRESHOLD = 25
_HEALTH_OHE_DIM = get_piecewise_dim(HEALTH_MIN, HEALTH_MAX_ENTITY, LINEAR_SQRT_THRESHOLD)
_BLOCK_OHE_DIM = get_piecewise_dim(BLOCK_MIN, BLOCK_MAX_ENTITY, LINEAR_SQRT_THRESHOLD)
_TOTAL_OHE_DIM = get_piecewise_dim(TOTAL_MIN, TOTAL_MAX, LINEAR_SQRT_THRESHOLD)


CHANNEL_LAYOUT_HEALTH_BLOCK, _alloc = make_channel_allocator()
OFF_HEALTH_BLOCK_HEALTH = _alloc("health_ohe", _HEALTH_OHE_DIM)
OFF_HEALTH_BLOCK_BLOCK = _alloc("block_ohe", _BLOCK_OHE_DIM)
OFF_HEALTH_BLOCK_TOTAL = _alloc("total_ohe", _TOTAL_OHE_DIM)
OFF_HEALTH_BLOCK_HEALTH_SCALAR = _alloc("health_scalar")
OFF_HEALTH_BLOCK_BLOCK_SCALAR = _alloc("block_scalar")
OFF_HEALTH_BLOCK_TOTAL_SCALAR = _alloc("total_scalar")

# The one width every combatant writes: the token rows' health/block columns
ENCODING_DIM_HEALTH_BLOCK = sum(w for _, w in CHANNEL_LAYOUT_HEALTH_BLOCK)

# Most nonzero health/block columns a row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_HEALTH_BLOCK = len(CHANNEL_LAYOUT_HEALTH_BLOCK)
