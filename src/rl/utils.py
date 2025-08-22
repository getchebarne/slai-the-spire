from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def encode_one_hot(
    value: int, value_min: int, value_max: int, device: torch.device
) -> torch.Tensor:
    value_clamp = max(min(value, value_max), value_min)
    bin_idx = value_clamp - value_min
    num_bins = value_max - value_min + 1
    return F.one_hot(torch.tensor(bin_idx, device=device), num_classes=num_bins).to(torch.float32)
