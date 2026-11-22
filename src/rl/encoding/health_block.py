import numpy as np

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import encode_piecewise_into
from src.rl.encoding.scalars import get_piecewise_dim


_HEALTH_MIN = 0
_BLOCK_MIN = 0
_TOTAL_MIN = _HEALTH_MIN + _BLOCK_MIN
_TOTAL_MAX = HEALTH_MAX_ENTITY + BLOCK_MAX_ENTITY
_LINEAR_SQRT_THRESHOLD = 25
_HEALTH_OHE_DIM = get_piecewise_dim(_HEALTH_MIN, HEALTH_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_BLOCK_OHE_DIM = get_piecewise_dim(_BLOCK_MIN, BLOCK_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_TOTAL_OHE_DIM = get_piecewise_dim(_TOTAL_MIN, _TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)


_CHANNEL_LAYOUT_HEALTH_BLOCK, _alloc = make_channel_allocator()
_OFF_HEALTH_BLOCK_HEALTH = _alloc("health_ohe", _HEALTH_OHE_DIM, 2)
_OFF_HEALTH_BLOCK_BLOCK = _alloc("block_ohe", _BLOCK_OHE_DIM, 2)
_OFF_HEALTH_BLOCK_TOTAL = _alloc("total_ohe", _TOTAL_OHE_DIM, 2)
_OFF_HEALTH_BLOCK_HEALTH_SCALAR = _alloc("health_scalar")
_OFF_HEALTH_BLOCK_BLOCK_SCALAR = _alloc("block_scalar")
_OFF_HEALTH_BLOCK_TOTAL_SCALAR = _alloc("total_scalar")

# The one width every combatant writes: the token rows' health/block columns
ENCODING_DIM_HEALTH_BLOCK = sum(w for _, w, _ in _CHANNEL_LAYOUT_HEALTH_BLOCK)

# Most nonzero health/block columns a row holds: one per one-hot or scalar, two per two-hot
ENCODING_DIM_SPARSE_HEALTH_BLOCK = sum(nonzeros for _, _, nonzeros in _CHANNEL_LAYOUT_HEALTH_BLOCK)


def encode_health_block_into(health: int, block: int, np_out: np.ndarray) -> None:
    """Write a combatant's health, block and their sum, each bucketed and as a scalar.

    Parameters
    ----------
    health : int
        The combatant's health.
    block : int
        The combatant's block.
    np_out : np.ndarray
        (ENCODING_DIM_HEALTH_BLOCK,) float32. The shared block's health/block range, written in
        place.
    """
    total = health + block

    encode_piecewise_into(
        health,
        _HEALTH_MIN,
        HEALTH_MAX_ENTITY,
        _LINEAR_SQRT_THRESHOLD,
        np_out[_OFF_HEALTH_BLOCK_HEALTH:][:_HEALTH_OHE_DIM],
    )
    encode_piecewise_into(
        block,
        _BLOCK_MIN,
        BLOCK_MAX_ENTITY,
        _LINEAR_SQRT_THRESHOLD,
        np_out[_OFF_HEALTH_BLOCK_BLOCK:][:_BLOCK_OHE_DIM],
    )
    encode_piecewise_into(
        total,
        _TOTAL_MIN,
        _TOTAL_MAX,
        _LINEAR_SQRT_THRESHOLD,
        np_out[_OFF_HEALTH_BLOCK_TOTAL:][:_TOTAL_OHE_DIM],
    )

    # Scalars
    np_out[_OFF_HEALTH_BLOCK_HEALTH_SCALAR] = min(health / HEALTH_MAX_ENTITY, 1.0)
    np_out[_OFF_HEALTH_BLOCK_BLOCK_SCALAR] = min(block / BLOCK_MAX_ENTITY, 1.0)
    np_out[_OFF_HEALTH_BLOCK_TOTAL_SCALAR] = min(total / _TOTAL_MAX, 1.0)
