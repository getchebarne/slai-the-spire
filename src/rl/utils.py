from typing import Any

import torch
import torch.nn as nn
import yaml


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
