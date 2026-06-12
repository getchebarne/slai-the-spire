import math
import typing
from typing import Any

import torch
import torch.nn as nn
import yaml


def get_piecewise_dim(val_min: int, val_max: int, threshold: int) -> int:
    """Dimension of a piecewise linear-sqrt one-hot over [val_min, val_max]"""
    if val_max <= threshold:
        # Pure linear, no square root piece
        return val_max - val_min + 1

    # Linear piece: val_min..threshold inclusive
    dim_linear = threshold - val_min + 1

    # Square root piece: floor(sqrt(threshold+1))..floor(sqrt(val_max))
    dim_sqrt = int(math.sqrt(val_max)) - int(math.sqrt(threshold))
    return dim_linear + dim_sqrt


def get_piecewise_bucket(value: int, val_min: int, val_max: int, threshold: int) -> int:
    """Bucket index of `value` in a piecewise linear-sqrt one-hot"""
    # Clamp
    value = min(max(value, val_min), val_max)
    if value <= threshold:
        return value - val_min

    sqrt_threshold = int(math.sqrt(threshold))
    sqrt_value = int(math.sqrt(value))
    return (threshold - val_min + 1) + (sqrt_value - sqrt_threshold - 1)


def get_sqrt_norm(value: float, cap: float) -> float:
    """Sign-preserving sqrt-compressed normalization into [-1, 1]

    Keeps resolution on small magnitudes while compressing large ones,
    matching the sqrt one-hot buckets.
    """
    clamped = min(max(value, -cap), cap)
    return math.copysign(math.sqrt(abs(clamped)) / math.sqrt(cap), clamped)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    if "num_episodes" in config:
        config["num_episodes"] = int(config["num_episodes"])
    if "num_iterations" in config:
        config["num_iterations"] = int(config["num_iterations"])
    if "buffer_size" in config:
        config["buffer_size"] = int(config["buffer_size"])

    return config


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())
