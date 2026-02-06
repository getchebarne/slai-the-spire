"""
Shared health/block encoding for monsters and character.

Uses piecewise linear-sqrt encoding with unified bounds so that
"42 HP with 5 block" produces the exact same encoding regardless
of whether it's on the character or a monster.
"""

import math

import numpy as np

from src.game.factory.monster.the_guardian import _HEALTH_MAX_ASC_9  # TODO: queries


# Unified bounds (max of character and monster ranges)
_HEALTH_MIN = 1
_HEALTH_MAX = _HEALTH_MAX_ASC_9  # 250 (covers both character 70 and monster 250)
_BLOCK_MIN = 0
_BLOCK_MAX = 20  # Same for both

# Piecewise linear-sqrt threshold (shared with monster.py)
_LINEAR_SQRT_THRESHOLD = 18


def _get_piecewise_dim(min_val: int, max_val: int, threshold: int) -> int:
    """Calculate dimension for piecewise linear-sqrt encoding."""
    if max_val <= threshold:
        return max_val - min_val + 1  # Pure linear
    # Linear part: min_val to threshold
    linear_dim = threshold - min_val + 1
    # Sqrt part: floor(sqrt(threshold+1)) to floor(sqrt(max_val))
    sqrt_dim = int(math.sqrt(max_val)) - int(math.sqrt(threshold))
    return linear_dim + sqrt_dim


def _get_piecewise_bucket(value: int, min_val: int, threshold: int) -> int:
    """Get bucket index for piecewise linear-sqrt encoding."""
    value = max(value, min_val)  # Clamp to min
    if value <= threshold:
        return value - min_val
    # Above threshold: use sqrt buckets
    sqrt_threshold = int(math.sqrt(threshold))
    sqrt_value = int(math.sqrt(value))
    return (threshold - min_val + 1) + (sqrt_value - sqrt_threshold - 1)


# Pre-computed dimensions
_HEALTH_DIM = _get_piecewise_dim(_HEALTH_MIN, _HEALTH_MAX, _LINEAR_SQRT_THRESHOLD)
_BLOCK_DIM = _get_piecewise_dim(_BLOCK_MIN, _BLOCK_MAX, _LINEAR_SQRT_THRESHOLD)
_HP_BLOCK_DIM = _get_piecewise_dim(
    _HEALTH_MIN + _BLOCK_MIN, _HEALTH_MAX + _BLOCK_MAX, _LINEAR_SQRT_THRESHOLD
)

# Total dimension: 3 piecewise OHEs + 3 scalars
_ENCODING_DIM = _HEALTH_DIM + _BLOCK_DIM + _HP_BLOCK_DIM + 3


def get_encoding_dim_health_block() -> int:
    """Return the dimension of the shared health/block encoding."""
    return _ENCODING_DIM


def encode_health_block_into(out: np.ndarray, health: int, block: int) -> None:
    """Encode health and block into a pre-allocated numpy array.

    Args:
        out: Pre-allocated array of shape (dim_health_block,), filled with zeros.
        health: Current health value.
        block: Current block value.
    """
    pos = 0

    # Health piecewise one-hot
    health_bucket = _get_piecewise_bucket(health, _HEALTH_MIN, _LINEAR_SQRT_THRESHOLD)
    out[pos + health_bucket] = 1.0
    pos += _HEALTH_DIM

    # Block piecewise one-hot
    block_bucket = _get_piecewise_bucket(block, _BLOCK_MIN, _LINEAR_SQRT_THRESHOLD)
    out[pos + block_bucket] = 1.0
    pos += _BLOCK_DIM

    # HP+Block piecewise one-hot
    hp_block = health + block
    hp_block_bucket = _get_piecewise_bucket(
        hp_block, _HEALTH_MIN + _BLOCK_MIN, _LINEAR_SQRT_THRESHOLD
    )
    out[pos + hp_block_bucket] = 1.0
    pos += _HP_BLOCK_DIM

    # Scalars
    out[pos] = health / _HEALTH_MAX
    out[pos + 1] = block / _BLOCK_MAX
    out[pos + 2] = hp_block / (_HEALTH_MAX + _BLOCK_MAX)
