from typing import Any

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    config["num_episodes"] = int(config["num_episodes"])
    config["buffer_size"] = int(config["buffer_size"])

    return config


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())
