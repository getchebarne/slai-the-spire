import warnings
from dataclasses import dataclass
from enum import Enum
from enum import IntEnum
from typing import NamedTuple

import torch
from slai import ACTION_SPEC_REGISTRY
from slai import ActionType
from slai import CandidatePoolDeck
from slai import CandidatePoolDiscover
from slai import CandidatePoolEventRollCard
from slai import CandidatePoolEventRollPotion
from slai import CandidatePoolEventRollRelic
from slai import CandidatePoolHand
from slai import CandidatePoolPileDiscard
from slai import CandidatePoolPileDraw
from slai import CandidatePoolPileExhaust
from slai import members
from tensordict import TensorDict
from tensordict import tensorclass


class SliceKind(Enum):
    CARD_HAND = 0
    CARD_DRAW = 1
    CARD_DISCARD = 2
    CARD_EXHAUST = 3
    CARD_DECK = 4
    CARD_DISCOVER = 5
    CARD_REWARD = 6
    CARD_SHOP = 7
    RELIC_OWNED = 8
    RELIC_REWARD = 9
    RELIC_SHOP = 10
    POTION_OWNED = 11
    POTION_REWARD = 12
    POTION_SHOP = 13
    MONSTERS = 14
    EVENT_OPTIONS = 15
    CHARACTER = 16
    ROOMS = 17
    CARD_STASIS = 18
    CARD_EVENT = 19
    POTION_EVENT = 20
    RELIC_EVENT = 21


class Slice(NamedTuple):
    kind: SliceKind
    size: int


# Slice kinds that have already warned about overflow — truncation is never silent
_WARNED_TRUNCATED: set[SliceKind] = set()


def clip_to_slice(entities: list, slice_: Slice) -> list:
    """Clip a pile to its slice cap with a warn-once (the shared warn+drop
    overflow policy: encoders drop the tail, masks drop the matching actions)."""
    if len(entities) > slice_.size:
        if slice_.kind not in _WARNED_TRUNCATED:
            _WARNED_TRUNCATED.add(slice_.kind)
            warnings.warn(
                f"{slice_.kind.name} pile of {len(entities)} entities truncated to"
                f" encoder cap ({slice_.size})"
            )
        return entities[: slice_.size]
    return entities


@tensorclass
class TPadded:
    x: torch.Tensor  # (B, S, D)
    mask: torch.Tensor  # (B, S) True = valid


@tensorclass
class TEffects:
    """Per-state pool of effect rows, flattened across entities.

    The per-entity axis is ragged (0-5 effects), so it is folded away: `segment`
    carries the owning token slot and the model sum-pools rows back into their
    owner. Pads are mask-False with segment 0, which aliases card slot 0 — only
    maskless misuse could read them."""

    x: torch.Tensor  # (B, MAX_EFFECTS_STATE, ENCODING_DIM_EFFECT) fp32
    segment: torch.Tensor  # (B, E) long — owner token slot; EFFECT_SEGMENT_PENDING for the halt
    mask: torch.Tensor  # (B, E) bool


@tensorclass
class TGameState:
    cards: TPadded
    relics: TPadded
    potions: TPadded
    monsters: TPadded
    event_options: TPadded
    character: TPadded
    effects: TEffects
    energy: torch.Tensor
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    room_node_idx: torch.Tensor  # (B, MAP_WIDTH) long
    room_mask: torch.Tensor  # (B, MAP_WIDTH) bool — next-row room non-padding mask
    map_meta: torch.Tensor
    screen: torch.Tensor
    ascension: torch.Tensor  # (B, ENCODING_DIM_ASCENSION) thermometer, bit i = level > i
    reward_meta: torch.Tensor
    shop_meta: torch.Tensor
    event_meta: torch.Tensor
    shop_card_prices: torch.Tensor
    shop_relic_prices: torch.Tensor
    shop_potion_prices: torch.Tensor


@tensorclass
class TEntityProjection:
    # EntityProjector output: one embedded TPadded per entity type; Core cats them (never here)
    cards: TPadded
    relics: TPadded
    potions: TPadded
    monsters: TPadded
    events: TPadded
    character: TPadded
    rooms: TPadded
    pending: torch.Tensor  # (B, dim) fp32 — pooled halt effect; zeros when no halt


# TODO: rethink `dict`
@dataclass
class TCoreOutput:
    global_: torch.Tensor
    pool: dict[SliceKind, torch.Tensor]  # selectable slices: refined[:, core.GLOBAL[kind]]


@tensorclass
class TMask:
    mask_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) — L1
    mask_action_idx: TensorDict  # {str(int(ActionType)): (B, size)} — L2, deduped
    mask_target_card: torch.Tensor  # (B, MAX_SIZE_HAND, MAX_MONSTERS) — L3
    mask_target_potion: torch.Tensor  # (B, MAX_POTION_SLOTS, MAX_MONSTERS) — L3


class Level(IntEnum):
    """Action-hierarchy levels — column indices into TAction's (B, len(Level)) tensors."""

    ACTION_TYPE = 0
    L1 = 1
    L2 = 2


@tensorclass
class TAction:
    idxs: torch.Tensor  # (B, len(Level)) long, per-Level column; -1 where the level didn't apply
    log_prob: torch.Tensor  # (B, len(Level)) float, 0 where N/A
    entropy: torch.Tensor  # (B, len(Level)) float, 0 where N/A
    kl_uniform: (
        torch.Tensor
    )  # (B, len(Level)) float, KL(uniform‖π) over legal actions, 0 where N/A


@tensorclass(shadow=True)  # `values` field shadows TensorDict.values()
class RolloutBuffer:
    """Columnar rollout (N = rollout_length * num_envs rows). The full-batch masks are
    stored once and row-sliced per minibatch; GAE returns/advantages are precomputed.
    Values/returns are per reward stream (K = len(REWARD_STREAMS)); the policy
    advantage is the per-stream advantages summed, then normalized."""

    game_state: TGameState  # (N, ...)
    mask_batch: TMask  # (N, ...) — row-sliced per minibatch (mask_batch[rows])
    idx_at: torch.Tensor  # (N,) recorded L1 ActionType pick
    idx_l1: torch.Tensor  # (N,) recorded L2 entity pick
    idx_l2: torch.Tensor  # (N,) recorded L3 monster pick (-1 if none)
    log_probs_old: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, K)
    returns: torch.Tensor  # (N, K)
    advantages: torch.Tensor  # (N, 1), summed over streams, normalized


# Sorted by int(): the engine's declaration-order discriminant, which is what
# masks/utils/heads index by. members() is alphabetical (dir()) and diverged from
# int() order when the engine appended action types — never use it positionally
ACTION_TYPE_BY_INT = sorted(members(ActionType), key=int)
NUM_ENGINE_ACTION_TYPES = len(ACTION_TYPE_BY_INT)


# Halting candidate pool -> the L2 slice its idx indexes
#
# The engine resolves every halt with one `EffectPendingResolve {idx}` over
# `pending.target.candidate_pool`. Masks, pointer nets and the shared-memory
# layout all need one static width per action type, so the policy splits that
# one engine type into a virtual type per pool and folds/unfolds at the boundary
# (masks.build_masks in, utils.engine_action out). At most one is ever legal:
# the engine holds a single effect_pending
PENDING_POOL_SLICE: dict[type, SliceKind] = {
    CandidatePoolHand: SliceKind.CARD_HAND,
    CandidatePoolPileDraw: SliceKind.CARD_DRAW,
    CandidatePoolPileDiscard: SliceKind.CARD_DISCARD,
    CandidatePoolPileExhaust: SliceKind.CARD_EXHAUST,
    CandidatePoolDeck: SliceKind.CARD_DECK,
    CandidatePoolDiscover: SliceKind.CARD_DISCOVER,
    CandidatePoolEventRollCard: SliceKind.CARD_EVENT,
    CandidatePoolEventRollRelic: SliceKind.RELIC_EVENT,
    CandidatePoolEventRollPotion: SliceKind.POTION_EVENT,
}
# Policy ints: engine ints 0..N-1 keep their values (so
# ACTION_TYPE_BY_INT.index(ActionType.X) still works), virtuals append
PENDING_POOL_INT: dict[type, int] = {
    pool: NUM_ENGINE_ACTION_TYPES + i for i, pool in enumerate(PENDING_POOL_SLICE)
}
PENDING_INTS: frozenset = frozenset(PENDING_POOL_INT.values())
NUM_ACTION_TYPES = NUM_ENGINE_ACTION_TYPES + len(PENDING_POOL_SLICE)

# Each index-taking action -> the L2 pool its selection indexes into, keyed by
# POLICY int (virtuals are not engine members). Terminal actions absent
# RewardTakeCard is arity-2 (idx_bundle, idx_card): its L2 pick is a slot in the
# bundle-flattened CARD_REWARD slice; masks/utils translate slot <-> (bundle, card)
# via per-state prefix sums. RewardSingingBowl points at a bundle via its first-card
# slot in the same slice. Only CardPlay/PotionUse have an L3 (monster target)
_ENGINE_POOL: dict[ActionType, SliceKind] = {
    ActionType.CardPlay: SliceKind.CARD_HAND,
    ActionType.EventOptionSelect: SliceKind.EVENT_OPTIONS,
    ActionType.PotionDiscard: SliceKind.POTION_OWNED,
    ActionType.PotionUse: SliceKind.POTION_OWNED,
    ActionType.RewardSingingBowl: SliceKind.CARD_REWARD,
    ActionType.RewardTakeCard: SliceKind.CARD_REWARD,
    ActionType.RewardTakePotion: SliceKind.POTION_REWARD,
    ActionType.RewardTakeRelic: SliceKind.RELIC_REWARD,
    ActionType.RoomSelect: SliceKind.ROOMS,
    ActionType.ShopBuyCard: SliceKind.CARD_SHOP,
    ActionType.ShopBuyPotion: SliceKind.POTION_SHOP,
    ActionType.ShopBuyRelic: SliceKind.RELIC_SHOP,
}
ACTION_TYPE_POOL: dict[int, SliceKind] = {
    **{int(at): pool for at, pool in _ENGINE_POOL.items()},
    **{PENDING_POOL_INT[pool]: kind for pool, kind in PENDING_POOL_SLICE.items()},
}

# Drift alarm: the engine's index-taking set must equal our engine-keyed entries
# plus the one resolver we split into the virtuals
_ACTION_TYPE_ARITY: list = [ACTION_SPEC_REGISTRY[m].arity for m in ACTION_TYPE_BY_INT]
_ENGINE_INDEX_TAKING = {
    at for at in range(NUM_ENGINE_ACTION_TYPES) if _ACTION_TYPE_ARITY[at] != (0, 0)
}
assert _ENGINE_INDEX_TAKING == {int(a) for a in _ENGINE_POOL} | {
    int(ActionType.EffectPendingResolve)
}, (
    "engine index-taking actions changed: _ENGINE_POOL + EffectPendingResolve "
    f"!= {sorted(_ENGINE_INDEX_TAKING)}"
)
