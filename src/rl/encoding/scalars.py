import math
from bisect import bisect_left

import numpy as np

from src.rl.constants import MAX_SCALED


# Each piecewise code's knots, by its (val_min, val_max, val_th), made on its first use
_KNOTS_OF_PIECEWISE: dict[tuple[int, int, int], list[int]] = {}


def _get_knots(val_min: int, val_max: int, val_th: int) -> list[int]:
    """Return the knots of a piecewise linear-sqrt code over [val_min, val_max].

    Every integer from `val_min` through min(`val_th`, `val_max`), then each square past
    `val_th` up to the first at or past `val_max`. They are made once per argument triple.

    Parameters
    ----------
    val_min : int
        The smallest value, the first knot.
    val_max : int
        The largest value.
    val_th : int
        The linear piece's last value.

    Returns
    -------
    list[int]
        The knots, increasing.
    """
    knots = _KNOTS_OF_PIECEWISE.get((val_min, val_max, val_th))
    if knots is None:
        knots = list(range(val_min, min(val_th, val_max) + 1))
        if val_max > val_th:
            root = math.isqrt(val_th)
            while root * root < val_max:
                root += 1
                knots.append(root * root)

        _KNOTS_OF_PIECEWISE[val_min, val_max, val_th] = knots

    return knots


def encode_piecewise_into(
    val: float, val_min: int, val_max: int, val_th: int, np_out: np.ndarray
) -> None:
    """Write `val` as a two-hot over the knots of a piecewise linear-sqrt code.

    A value on a knot sets 1.0 there. One between two adjacent knots a < val < b sets
    (b - val) / (b - a) at a's column and (val - a) / (b - a) at b's: the weights sum to 1,
    and the knots they weigh sum to the value.

    Parameters
    ----------
    val : float
        The value to write, clamped to [val_min, val_max].
    val_min : int
        The smallest value, the first knot.
    val_max : int
        The largest value.
    val_th : int
        The linear piece's last value: each integer from `val_min` through min(`val_th`,
        `val_max`) is a knot, then each square past it up to the first at or past `val_max`.
    np_out : np.ndarray
        (C,) float32, C at least get_piecewise_dim(val_min, val_max, val_th). The channel's
        columns, one per knot, written in place.
    """
    knots = _get_knots(val_min, val_max, val_th)

    # Clamp
    val = min(max(val, val_min), val_max)

    # The first knot at or past the val
    idx_knot = bisect_left(knots, val)

    knot_upper = knots[idx_knot]
    if knot_upper == val:
        np_out[idx_knot] = 1.0

        return

    knot_lower = knots[idx_knot - 1]
    np_out[idx_knot - 1] = (knot_upper - val) / (knot_upper - knot_lower)
    np_out[idx_knot] = (val - knot_lower) / (knot_upper - knot_lower)


def get_piecewise_dim(val_min: int, val_max: int, val_th: int) -> int:
    """Return the dimension of a piecewise linear-sqrt two-hot over [val_min, val_max].

    Parameters
    ----------
    val_min : int
        The smallest value.
    val_max : int
        The largest value.
    val_th : int
        The linear piece's last value, as in `encode_piecewise_into`.

    Returns
    -------
    int
        The number of knots.
    """
    return len(_get_knots(val_min, val_max, val_th))


def get_sqrt_norm(val: float, cap: float) -> float:
    """Normalize `val` into [-1, 1], sign-preserving and sqrt-compressed.

    Keeps resolution on small magnitudes while compressing large ones.

    Parameters
    ----------
    val : float
        The value to normalize.
    cap : float
        The magnitude that maps to 1; larger ones clamp to it.

    Returns
    -------
    float
        The square root of |val| / `cap`, with |val| clamped to `cap`, signed like `val`.
    """
    clamped = min(max(val, -cap), cap)

    return math.copysign(math.sqrt(abs(clamped)) / math.sqrt(cap), clamped)


def get_linear_norm(val: float, scale: float) -> float:
    """Normalize `val` linearly: `val` / `scale`, clipped to [-MAX_SCALED, MAX_SCALED].

    Parameters
    ----------
    val : float
        The value to normalize.
    scale : float
        The value of one unit of the channel, such as SCALE_HEALTH or SCALE_GOLD.

    Returns
    -------
    float
        `val` / `scale`, clipped to [-MAX_SCALED, MAX_SCALED].
    """
    return min(max(val / scale, -MAX_SCALED), MAX_SCALED)
