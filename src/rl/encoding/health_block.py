from enum import StrEnum
from enum import auto

import numpy as np

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.types import make_channel_allocator


_HEALTH_MIN = 1
_BLOCK_MIN = 0
_TOTAL_MIN = _HEALTH_MIN + _BLOCK_MIN
_TOTAL_MAX = HEALTH_MAX_ENTITY + BLOCK_MAX_ENTITY
_LINEAR_SQRT_THRESHOLD = 25
_HEALTH_OHE_DIM = get_piecewise_dim(_HEALTH_MIN, HEALTH_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_BLOCK_OHE_DIM = get_piecewise_dim(_BLOCK_MIN, BLOCK_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_TOTAL_OHE_DIM = get_piecewise_dim(_TOTAL_MIN, _TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)


class HealthBlockChannel(StrEnum):
    HEALTH_OHE = auto()
    BLOCK_OHE = auto()
    TOTAL_OHE = auto()
    HEALTH_SCALAR = auto()
    BLOCK_SCALAR = auto()
    TOTAL_SCALAR = auto()


CHANNEL_LAYOUT_HEALTH_BLOCK, _alloc = make_channel_allocator()
OFF_HEALTH_BLOCK_HEALTH = _alloc(HealthBlockChannel.HEALTH_OHE, _HEALTH_OHE_DIM)
OFF_HEALTH_BLOCK_BLOCK = _alloc(HealthBlockChannel.BLOCK_OHE, _BLOCK_OHE_DIM)
OFF_HEALTH_BLOCK_TOTAL = _alloc(HealthBlockChannel.TOTAL_OHE, _TOTAL_OHE_DIM)
OFF_HEALTH_BLOCK_HEALTH_SCALAR = _alloc(HealthBlockChannel.HEALTH_SCALAR)
OFF_HEALTH_BLOCK_BLOCK_SCALAR = _alloc(HealthBlockChannel.BLOCK_SCALAR)
OFF_HEALTH_BLOCK_TOTAL_SCALAR = _alloc(HealthBlockChannel.TOTAL_SCALAR)

# The one width every combatant writes: the token rows' health/block columns
ENCODING_DIM_HEALTH_BLOCK = sum(w for _, w in CHANNEL_LAYOUT_HEALTH_BLOCK)

# Most nonzero health/block columns a row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_HEALTH_BLOCK = len(CHANNEL_LAYOUT_HEALTH_BLOCK)


def encode_health_block_into(health: int, block: int, out: np.ndarray) -> None:
    total = health + block

    health_bucket = get_piecewise_bucket(
        health, _HEALTH_MIN, HEALTH_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_HEALTH + health_bucket] = 1.0
    block_bucket = get_piecewise_bucket(
        block, _BLOCK_MIN, BLOCK_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_BLOCK + block_bucket] = 1.0
    total_bucket = get_piecewise_bucket(total, _TOTAL_MIN, _TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)
    out[OFF_HEALTH_BLOCK_TOTAL + total_bucket] = 1.0

    # Scalars
    out[OFF_HEALTH_BLOCK_HEALTH_SCALAR] = min(health / HEALTH_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_BLOCK_SCALAR] = min(block / BLOCK_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_TOTAL_SCALAR] = min(total / _TOTAL_MAX, 1.0)
