import math


def get_piecewise_bucket(value: int, val_min: int, val_max: int, threshold: int) -> int:
    """Return the bucket index of `value` in a piecewise linear-sqrt one-hot.

    Parameters
    ----------
    value : int
        The value to bucket, clamped to [val_min, val_max].
    val_min : int
        The smallest value, in bucket 0.
    val_max : int
        The largest value.
    threshold : int
        The linear piece's last value: each value up to it has its own bucket, and the larger
        ones share one bucket per floor(sqrt(value)).

    Returns
    -------
    int
        The bucket index.
    """

    # Clamp
    value = min(max(value, val_min), val_max)

    # Linear piece: val_min..threshold inclusive
    if value <= threshold:
        return value - val_min

    # Square root piece: floor(sqrt(threshold+1))..floor(sqrt(val_max))
    sqrt_start = int(math.sqrt(threshold + 1))
    sqrt_value = int(math.sqrt(value))
    return (threshold - val_min + 1) + (sqrt_value - sqrt_start)


def get_piecewise_dim(val_min: int, val_max: int, threshold: int) -> int:
    """Return the dimension of a piecewise linear-sqrt one-hot over [val_min, val_max].

    Parameters
    ----------
    val_min : int
        The smallest value.
    val_max : int
        The largest value.
    threshold : int
        The linear piece's last value, as in `get_piecewise_bucket`.

    Returns
    -------
    int
        The number of buckets.
    """
    return get_piecewise_bucket(val_max, val_min, val_max, threshold) + 1


def get_sqrt_norm(value: float, cap: float) -> float:
    """Normalize `value` into [-1, 1], sign-preserving and sqrt-compressed.

    Keeps resolution on small magnitudes while compressing large ones, matching the sqrt one-hot
    buckets.

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
