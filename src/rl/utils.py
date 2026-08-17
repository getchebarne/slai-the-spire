import math
from bisect import bisect_right
from typing import Any

import torch
import torch.nn as nn
import yaml
from slai import Action
from slai import ActionType
from slai import GameState

from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import RolloutBuffer
from src.rl.types import TAction


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


# ---- Reward-card slot translation ----
# Reward.cards is ragged bundles (combat rolls <=4 cards, The Library rolls 20),
# so the CARD_REWARD slice is the bundle-flattened concatenation and the L1 slot
# <-> engine (idx_bundle[, idx_card]) translation runs through prefix sums. This
# is the single home for both directions: masks.py folds legal actions forward,
# action_from_actiontype unfolds picks back, and the fuzz oracle reuses both.


def reward_card_prefix(state: GameState) -> list[int]:
    """Cumulative token offsets of the reward-card bundles: bundle b's cards
    occupy CARD_REWARD slots [prefix[b], prefix[b+1])."""
    prefix = [0]
    reward = state.reward
    if reward is not None:
        for bundle in reward.cards:
            prefix.append(prefix[-1] + len(bundle))
    return prefix


def reward_slot(prefix: list[int], action_type: ActionType, idxs) -> int:
    """Engine (idx_bundle[, idx_card]) -> flat CARD_REWARD slot. RewardSingingBowl
    points at a bundle via its first-card token."""
    if action_type == ActionType.RewardTakeCard:
        return prefix[idxs[0]] + idxs[1]
    return prefix[idxs[0]]


def reward_idxs(prefix: list[int], action_type: ActionType, slot: int) -> list[int]:
    """Flat CARD_REWARD slot -> engine idxs (inverse of reward_slot)."""
    bundle = bisect_right(prefix, slot) - 1
    if action_type == ActionType.RewardTakeCard:
        return [bundle, slot - prefix[bundle]]
    return [bundle]


_REWARD_CARD_ACTIONS = frozenset((ActionType.RewardTakeCard, ActionType.RewardSingingBowl))


def engine_action(
    at_int: int, selection_index: int, target_index: int, state: GameState
) -> Action:
    """Build a `slai.Action` from the raw hierarchy picks [ActionType, L1, L2]
    (L2 = -1 when the action takes no monster target).

    `state` is the game state the mask was built from: the reward-card actions'
    L1 slot is a position in the bundle-flattened CARD_REWARD slice and must be
    split back into engine (idx_bundle, idx_card) via the same prefix sums."""
    action_type = ACTION_TYPE_BY_INT[at_int]
    if action_type not in ACTION_TYPE_POOL:
        idxs = []
    elif action_type in _REWARD_CARD_ACTIONS:
        idxs = reward_idxs(reward_card_prefix(state), action_type, selection_index)
    elif target_index >= 0:  # CardPlay / PotionUse with a monster target
        idxs = [selection_index, target_index]
    else:
        idxs = [selection_index]
    return Action(action_type, idxs)


def action_from_actiontype(t_action: TAction, i: int, state: GameState) -> Action:
    """Decode row `i` of a batched `TAction` into a `slai.Action`."""
    at_int, selection_index, target_index = t_action.idxs[i].tolist()
    return engine_action(at_int, selection_index, target_index, state)


def shuffle_rollout_buffer(buffer: RolloutBuffer, device: torch.device) -> RolloutBuffer:
    """Return the buffer with every row permuted by one shared random permutation, so each
    epoch's minibatches are cheap contiguous slice views (tensorclass indexing is ~25 ms/call)."""
    perm = torch.randperm(len(buffer), device=device)
    return buffer[perm]
