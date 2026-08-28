import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import yaml
from slai import Action
from slai import ActionType
from slai import GameState

from src.rl.constants import REWARD_REGION_OFFSETS
from src.rl.constants import REWARD_REGION_WIDTHS
from src.rl.constants import REWARD_SLOT_TO_BUNDLE
from src.rl.constants import REWARD_SLOT_TO_CARD
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import PENDING_INTS
from src.rl.types import PENDING_POOL_INT
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
# Reward.cards is ragged bundles (combat rolls <=4 cards, The Library rolls 20).
# Each bundle owns a fixed CARD_REWARD region (constants.REWARD_REGION_WIDTHS),
# so the L1 slot <-> engine (idx_bundle[, idx_card]) translation is static
# tables. This is the single home for both directions: masks.py folds legal
# actions forward, action_from_actiontype unfolds picks back, and the fuzz
# oracle reuses both


def reward_slot(action_type: ActionType, idxs: Sequence[int]) -> int:
    """Engine (idx_bundle[, idx_card]) -> CARD_REWARD slot. RewardSingingBowl
    points at a bundle via its first-card token."""
    if action_type == ActionType.RewardTakeCard:
        assert idxs[1] < REWARD_REGION_WIDTHS[idxs[0]], "reward card exceeds its region"
        return REWARD_REGION_OFFSETS[idxs[0]] + idxs[1]
    return REWARD_REGION_OFFSETS[idxs[0]]


def _reward_idxs(action_type: ActionType, slot: int) -> list[int]:
    """CARD_REWARD slot -> engine idxs (inverse of reward_slot)."""
    if action_type == ActionType.RewardTakeCard:
        return [REWARD_SLOT_TO_BUNDLE[slot], REWARD_SLOT_TO_CARD[slot]]
    return [REWARD_SLOT_TO_BUNDLE[slot]]


_REWARD_CARD_ACTIONS = frozenset((ActionType.RewardTakeCard, ActionType.RewardSingingBowl))


# ---- Pending-halt fold/unfold ----
# The engine resolves every halt with one EffectPendingResolve; the policy
# splits it per candidate pool (types.PENDING_POOL_SLICE) so each type keeps a
# single static L2 width. Both directions live here, next to the reward-card
# translation, for the same reason: masks folds forward, engine_action unfolds


def pending_virtual_int(state: GameState) -> int | None:
    """The virtual type for this state's halt, or None when nothing is pending.

    Resolve once per state, not once per legal action: every read here is an FFI
    getter that deep-clones, and a deck halt offers 45 of them."""
    pending = state.effect_pending
    if pending is None:
        return None
    if pending.target is None:
        raise RuntimeError("pending effect carries no Resolve target")
    pool = pending.target.candidate_pool
    virtual = PENDING_POOL_INT.get(pool)
    if virtual is None:
        raise ValueError(f"halt over an unmapped candidate pool: {pool}")
    return virtual


def virtual_action_int(action: Action, state: GameState) -> int:
    """Engine action -> policy action-type int. Only the halt resolver moves;
    every other type keeps its engine discriminant."""
    if action.action_type != ActionType.EffectPendingResolve:
        return int(action.action_type)
    virtual = pending_virtual_int(state)
    if virtual is None:
        raise RuntimeError("EffectPendingResolve legal with no pending effect")
    return virtual


def engine_action(
    at_int: int, selection_index: int, target_index: int, state: GameState
) -> Action:
    """Build a `slai.Action` from the raw hierarchy picks [ActionType, L1, L2]
    (L2 = -1 when the action takes no monster target).

    `state` is the game state the mask was built from: the reward-card actions'
    L1 slot is a position in the concatenated per-bundle reward slices and is
    split back into engine (idx_bundle, idx_card) via the static region tables."""
    if at_int in PENDING_INTS:  # any virtual pending type folds back to the one resolver
        return Action(ActionType.EffectPendingResolve, [selection_index])
    action_type = ACTION_TYPE_BY_INT[at_int]
    if at_int not in ACTION_TYPE_POOL:
        idxs = []
    elif action_type in _REWARD_CARD_ACTIONS:
        idxs = _reward_idxs(action_type, selection_index)
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
