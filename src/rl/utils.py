import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from slai import Action
from slai import ActionType
from slai import GameEnv
from slai import GameState

from src.rl.constants import FAST_MODE
from src.rl.constants import NEOW


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    if "num_iterations" in config:
        config["num_iterations"] = int(config["num_iterations"])

    return config


def set_rng_seeds(seed: int) -> None:
    """Seed the global generators of random, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())


def new_env(ascension: int) -> GameEnv:
    return GameEnv(ascension=ascension, fast_mode=FAST_MODE, neow=NEOW)


# Dominant actions the policy never chooses: step_scripted takes them as soon as they're
# legal. Taking a reward's gold never hurts
_SCRIPTED_ACTION_TYPES = frozenset({ActionType.RewardTakeGold, ActionType.RewardTakeGoldStolen})


def step_scripted(env: GameEnv, action: Action) -> tuple[GameState, bool]:
    """`env.step(action)`, then the scripted actions until none is legal, all inside
    this one policy step."""
    state, terminated = env.step(action)
    while not terminated:
        scripted = [a for a in env.get_legal_actions() if a.action_type in _SCRIPTED_ACTION_TYPES]
        if not scripted:
            break
        state, terminated = env.step(scripted[0])
    return state, terminated
