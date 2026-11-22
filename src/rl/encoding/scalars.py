import math
from bisect import bisect_left

import numpy as np


# Each piecewise code's knots, by its (val_min, val_max, threshold), made on its first use
_KNOTS_OF_PIECEWISE: dict[tuple[int, int, int], list[int]] = {}


def _get_knots(val_min: int, val_max: int, threshold: int) -> list[int]:
    """Return the knots of a piecewise linear-sqrt code over [val_min, val_max].

    Every integer from `val_min` through min(`threshold`, `val_max`), then each square past
    `threshold` up to the first at or past `val_max`. They are made once per argument triple.

    Parameters
    ----------
    val_min : int
        The smallest value, the first knot.
    val_max : int
        The largest value.
    threshold : int
        The linear piece's last value.

    Returns
    -------
    list[int]
        The knots, increasing.
    """
    knots = _KNOTS_OF_PIECEWISE.get((val_min, val_max, threshold))
    if knots is None:
        knots = list(range(val_min, min(threshold, val_max) + 1))
        if val_max > threshold:
            root = math.isqrt(threshold)
            while root * root < val_max:
                root += 1
                knots.append(root * root)
        _KNOTS_OF_PIECEWISE[val_min, val_max, threshold] = knots

    return knots


def encode_piecewise_into(
    value: float, val_min: int, val_max: int, threshold: int, np_out: np.ndarray
) -> None:
    """Write `value` as a two-hot over the knots of a piecewise linear-sqrt code.

    A value on a knot sets 1.0 there. One between two adjacent knots a < value < b sets
    (b - value) / (b - a) at a's column and (value - a) / (b - a) at b's: the weights sum to 1,
    and the knots they weigh sum to the value.

    Parameters
    ----------
    value : float
        The value to write, clamped to [val_min, val_max].
    val_min : int
        The smallest value, the first knot.
    val_max : int
        The largest value.
    threshold : int
        The linear piece's last value: each integer from `val_min` through min(`threshold`,
        `val_max`) is a knot, then each square past it up to the first at or past `val_max`.
    np_out : np.ndarray
        (C,) float32, C at least get_piecewise_dim(val_min, val_max, threshold). The channel's
        columns, one per knot, written in place.
    """
    knots = _get_knots(val_min, val_max, threshold)

    # Clamp
    value = min(max(value, val_min), val_max)

    # The first knot at or past the value
    idx_knot = bisect_left(knots, value)
    knot_upper = knots[idx_knot]
    if knot_upper == value:
        np_out[idx_knot] = 1.0
    else:
        knot_lower = knots[idx_knot - 1]
        np_out[idx_knot - 1] = (knot_upper - value) / (knot_upper - knot_lower)
        np_out[idx_knot] = (value - knot_lower) / (knot_upper - knot_lower)


def get_piecewise_dim(val_min: int, val_max: int, threshold: int) -> int:
    """Return the dimension of a piecewise linear-sqrt two-hot over [val_min, val_max].

    Parameters
    ----------
    val_min : int
        The smallest value.
    val_max : int
        The largest value.
    threshold : int
        The linear piece's last value, as in `encode_piecewise_into`.

    Returns
    -------
    int
        The number of knots.
    """
    return len(_get_knots(val_min, val_max, threshold))


def get_sqrt_norm(value: float, cap: float) -> float:
    """Normalize `value` into [-1, 1], sign-preserving and sqrt-compressed.

    Keeps resolution on small magnitudes while compressing large ones.

    Parameters
    ----------
    value : float
        The value to normalize.
    cap : float
        The magnitude that maps to 1; larger ones clamp to it.

    Returns
    -------
    float
        The square root of |value| / `cap`, with |value| clamped to `cap`, signed like `value`.
    """
    clamped = min(max(value, -cap), cap)

    return math.copysign(math.sqrt(abs(clamped)) / math.sqrt(cap), clamped)
