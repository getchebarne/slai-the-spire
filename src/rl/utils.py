import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from slai import Action
from slai import ActionType
from slai import Card
from slai import GameState

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import PENDING_INTS
from src.rl.types import PENDING_POOL_INT
from src.rl.types import RolloutBuffer
from src.rl.types import SliceKind
from src.rl.types import X_Action


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


# ---- Pending-halt fold/unfold ----
# The engine resolves every halt with one EffectPendingResolve; the policy
# splits it per candidate pool (types.PENDING_POOL_SLICE) so each halt scores
# exactly its pool's kind. Both directions live here: masks folds forward,
# engine_action unfolds


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


def card_pile_draw_offset(card_pile_draw: list[Card]) -> int:
    """Engine index of the draw pile's first token. The tokens are the pile's top
    MAX_SIZE_CARD_PILE_DRAW cards in engine order (bottom first), so token rank r
    is engine index offset + r, and a card below the offset has no token."""
    return max(0, len(card_pile_draw) - MAX_SIZE_CARD_PILE_DRAW)


# The kinds each policy action type may pick from (the decode assert)
_POOL_KIND_VALUES: dict[int, frozenset[int]] = {
    at: frozenset(kind for kind in kinds) for at, kinds in ACTION_TYPE_POOL.items()
}


def engine_action(
    at_int: int,
    kind: int,
    rank: int,
    target: int,
    room_idxs_row: np.ndarray,
    state: GameState,
) -> Action:
    """Build a `slai.Action` from the raw hierarchy picks [ActionType, kind, rank,
    target] (kind/rank = -1 when the type takes no pick, target = -1 when it takes
    no monster target).

    A pick names the rank-th token of its kind, and that rank is the engine index —
    except for a room, which the engine names by map column: the rank-th reachable
    column of `room_idxs_row`, the state's rooms row."""
    if at_int not in ACTION_TYPE_POOL:
        return Action(ACTION_TYPE_BY_INT[at_int], [])

    assert kind in _POOL_KIND_VALUES[at_int], f"pick kind {kind} is not in pool {at_int}"
    idx = rank
    if kind == SliceKind.ROOMS:
        idx = int(np.flatnonzero(room_idxs_row >= 0)[rank])
    elif kind == SliceKind.CARD_PILE_DRAW:
        # The draw pile's tokens are its top cards: rank 0 sits at the pile's offset
        combat = state.combat
        assert combat is not None, "a draw-pile pick outside combat"
        idx = rank + card_pile_draw_offset(combat.card_pile_draw)

    if at_int in PENDING_INTS:  # any virtual pending type folds back to the one resolver
        return Action(ActionType.EffectPendingResolve, [idx])
    action_type = ACTION_TYPE_BY_INT[at_int]
    if action_type is ActionType.RewardTakeCard:
        return Action(action_type, [kind - SliceKind.CARD_REWARD_0, idx])
    if action_type is ActionType.RewardSingingBowl:
        return Action(action_type, [kind - SliceKind.CARD_REWARD_0])
    if target >= 0:  # CardPlay / PotionUse with a monster target
        return Action(action_type, [idx, target])
    return Action(action_type, [idx])


def action_from_actiontype(
    x_action: X_Action, i: int, room_idxs_row: np.ndarray, state: GameState
) -> Action:
    """Decode row `i` of a batched `X_Action` into a `slai.Action`; `room_idxs_row`
    is that state's rooms row."""
    return engine_action(
        int(x_action.action_type[i]),
        int(x_action.kind[i]),
        int(x_action.rank[i]),
        int(x_action.target[i]),
        room_idxs_row,
        state,
    )


def shuffle_rollout_buffer(buffer: RolloutBuffer, device: torch.device) -> RolloutBuffer:
    """Return the buffer with every row permuted by one shared random permutation, so each
    epoch's minibatches are cheap contiguous slice views (tensorclass indexing is ~25 ms/call)."""
    perm = torch.randperm(len(buffer), device=device)
    return buffer[perm]
