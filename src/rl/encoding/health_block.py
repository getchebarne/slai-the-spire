import numpy as np

from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim


_HEALTH_MIN = 1
_BLOCK_MIN = 0
_LINEAR_SQRT_THRESHOLD = 25


def get_encoding_dim_health_block(health_max: int, block_max: int) -> int:
    """Return the dimension of the shared health/block encoding."""
    return (
        get_piecewise_dim(_HEALTH_MIN, health_max, _LINEAR_SQRT_THRESHOLD)
        + get_piecewise_dim(_BLOCK_MIN, block_max, _LINEAR_SQRT_THRESHOLD)
        + get_piecewise_dim(
            _HEALTH_MIN + _BLOCK_MIN, health_max + block_max, _LINEAR_SQRT_THRESHOLD
        )
        + 1  # Health scalar
        + 1  # Block scalar
        + 1  # Health + block scalar
    )


def encode_health_block_into(
    health: int, block: int, health_max: int, block_max: int, pos: int, out: np.ndarray
) -> int:
    health_block_min = _HEALTH_MIN + _BLOCK_MIN
    health_block_max = health_max + block_max
    health_dim = get_piecewise_dim(_HEALTH_MIN, health_max, _LINEAR_SQRT_THRESHOLD)
    block_dim = get_piecewise_dim(_BLOCK_MIN, block_max, _LINEAR_SQRT_THRESHOLD)
    health_block_dim = get_piecewise_dim(
        health_block_min, health_block_max, _LINEAR_SQRT_THRESHOLD
    )

    # Health piecewise one-hot
    health_bucket = get_piecewise_bucket(health, _HEALTH_MIN, health_max, _LINEAR_SQRT_THRESHOLD)
    out[pos + health_bucket] = 1.0
    pos += health_dim

    # Block piecewise one-hot
    block_bucket = get_piecewise_bucket(block, _BLOCK_MIN, block_max, _LINEAR_SQRT_THRESHOLD)
    out[pos + block_bucket] = 1.0
    pos += block_dim

    # Health + block piecewise one-hot
    health_block = health + block
    health_block_bucket = get_piecewise_bucket(
        health_block, health_block_min, health_block_max, _LINEAR_SQRT_THRESHOLD
    )
    out[pos + health_block_bucket] = 1.0
    pos += health_block_dim

    # Scalars (clamped to [0, 1] — block can exceed block_max, e.g. 60 block vs cap 20)
    out[pos] = min(health / health_max, 1.0)
    out[pos + 1] = min(block / block_max, 1.0)
    out[pos + 2] = min(health_block / health_block_max, 1.0)
    pos += 3

    return pos
