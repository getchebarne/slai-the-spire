import math


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
