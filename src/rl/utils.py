import random
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
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
    return GameEnv(ascension=ascension, fast_mode=FAST_MODE, neow=NEOW)


# Dominant actions the policy never chooses: step_scripted takes them as soon as they're
# legal. Taking a Reward's gold never hurts
_SCRIPTED_ACTION_TYPES = frozenset({ActionType.RewardTakeGold, ActionType.RewardTakeGoldStolen})


def step_scripted(env: GameEnv, action: Action) -> tuple[GameState, bool]:
    """Run `env.step(action)`, then the scripted actions until none is legal.

    All of them run inside this one policy step.

    Parameters
    ----------
    env : GameEnv
        The game to step.
    action : Action
        The policy's action.

    Returns
    -------
    game_state : GameState
        The state after the last action taken.
    terminated : bool
        Whether the game ended.
    """
    game_state, terminated = env.step(action)
    while not terminated:
        scripted = [a for a in env.get_legal_actions() if a.action_type in _SCRIPTED_ACTION_TYPES]
        if not scripted:
            break
        game_state, terminated = env.step(scripted[0])
    return game_state, terminated


def parent_alive() -> bool:
    """Return whether the process that started this one is still alive.

    Returns
    -------
    bool
        Whether the parent process is alive; False when this process has none.
    """
    parent = mp.parent_process()
    return parent is not None and parent.is_alive()
