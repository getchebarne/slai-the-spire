import random
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from slai import GameEnv

from src.rl.constants import NEOW


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    if "num_iterations" in config:
        config["num_iterations"] = int(config["num_iterations"])

    return config


def set_rng_seeds(seed: int) -> None:
    """Seed the global generators of random, numpy and torch.

    Parameters
    ----------
    seed : int
        The seed, taken modulo 2**32 for numpy.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())


def new_env(ascension: int) -> GameEnv:
    return GameEnv(ascension=ascension, neow=NEOW)


def parent_alive() -> bool:
    """Return whether the process that started this one is still alive.

    Returns
    -------
    bool
        Whether the parent process is alive; False when this process has none.
    """
    parent = mp.parent_process()
    return parent is not None and parent.is_alive()
