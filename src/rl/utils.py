from typing import Any

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    config["num_episodes"] = int(config["num_episodes"])

    if "buffer_size" in config:
        config["buffer_size"] = int(config["buffer_size"])

    return config


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())


def encode_one_hot_list(value: int, value_min: int, value_max: int) -> list[float]:
    value_clamp = max(min(value, value_max), value_min)
    value_clamp_offset = value_clamp - value_min
    ohe = [0.0] * (value_max - value_min + 1)
    ohe[value_clamp_offset] = 1.0

    return ohe
