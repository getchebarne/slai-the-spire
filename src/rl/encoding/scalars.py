import math


def get_piecewise_bucket(value: int, val_min: int, val_max: int, threshold: int) -> int:
    """Bucket index of `value` in a piecewise linear-sqrt one-hot"""

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
    """Dimension of a piecewise linear-sqrt one-hot over [val_min, val_max]"""
    return get_piecewise_bucket(val_max, val_min, val_max, threshold) + 1


def get_sqrt_norm(value: float, cap: float) -> float:
    """Sign-preserving sqrt-compressed normalization into [-1, 1]

    Keeps resolution on small magnitudes while compressing large ones,
    matching the sqrt one-hot buckets.
    """
    clamped = min(max(value, -cap), cap)
    return math.copysign(math.sqrt(abs(clamped)) / math.sqrt(cap), clamped)
