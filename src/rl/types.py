from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from enum import StrEnum

import torch
from slai import ACTION_SPEC_REGISTRY
from slai import ActionType
from slai import CandidatePool
from tensordict import tensorclass


class SliceKind(IntEnum):
    CARD_PILE_HAND = 0
    CARD_PILE_DRAW = 1
    CARD_PILE_DISCARD = 2
    CARD_PILE_EXHAUST = 3
    CARD_PILE_DECK = 4
    CARD_PILE_DISCOVER = 5
    CARD_REWARD_0 = 6
    CARD_REWARD_1 = 7
    CARD_REWARD_2 = 8
    CARD_REWARD_3 = 9
    CARD_REWARD_4 = 10
    CARD_SHOP = 11
    RELIC_OWNED = 12
    RELIC_REWARD = 13
    RELIC_SHOP = 14
    POTION_OWNED = 15
    POTION_REWARD = 16
    POTION_SHOP = 17
    MONSTERS = 18
    EVENT_OPTIONS = 19
    CHARACTER = 20
    ROOMS = 21
    CARD_PILE_STASIS = 22
    CARD_PILE_EVENT_ROLL = 23
    POTION_EVENT_ROLL = 24
    RELIC_EVENT_ROLL = 25
    CARD_PENDING_SELECTED = 26
    CARD_PLAY_QUEUE = 27
    CARD_PILE_NIGHTMARE = 28
    CARD_MATCH_FLIPPED = 29


def make_channel_allocator() -> tuple[list[tuple[StrEnum, int]], Callable[..., int]]:
    """A (layout, alloc) pair: alloc(channel, width=1) appends and returns the
    offset, so offsets, widths and the total all derive from one list."""
    layout: list[tuple[StrEnum, int]] = []

    def alloc(channel: StrEnum, width: int = 1) -> int:
        offset = sum(w for _, w in layout)
        layout.append((channel, width))
        return offset

    return layout, alloc


# Entity types that have already warned about overflow — truncation is never silent
class EntityKind(IntEnum):
    """The one entity-type order: X_GameState's entity fields, the model's block
    cat, and the owner type of an effect/modifier row all follow it."""

    CARD = 0
    RELIC = 1
    POTION = 2
    MONSTER = 3
    EVENT_OPTION = 4
    CHARACTER = 5


# Effect owner past the six types: the pending halt owns no token
OWNER_PENDING = len(EntityKind)


class EffectSource(IntEnum):
    """What applies the effect, and for relics on which trigger; stamped into the
    row by the effects encoder, outside the row cache."""

    CARD_PLAY = 0
    POTION_USE = 1
    EVENT_OPTION_SELECT = 2
    PENDING = 3

    # Monsters
    MONSTER_MOVE = 4
    MONSTER_MOVESET = 5

    # Relics
    RELIC_COMBAT_START = 6
    RELIC_TURN_START = 7
    RELIC_TURN_END = 8
    RELIC_COMBAT_END = 9
    RELIC_PICKUP = 10
    RELIC_REST = 11
    RELIC_COUNTER = 12

    # Cards, when discarded and when drawn
    CARD_DISCARD = 13
    CARD_DRAW = 14


@tensorclass
class X_Entity:
    """One entity type's tokens, packed to a valid prefix; `idxs[b, w]` is token
    w's SliceKind, -1 past the prefix (pad). W is the batch's longest packed
    sequence of the type — no fixed width anywhere but the transport's capacity."""

    x: torch.Tensor  # (B, W, D_type)
    x_slice_kind: torch.Tensor  # (B, W) int16 (long once the model owns the window)

    @property
    def mask(self) -> torch.Tensor:
        return self.x_slice_kind >= 0


@tensorclass
class X_Owned:
    """Per-state pool of effect rows, flattened across entities.

    The per-entity axis is ragged (0-5 effects), so it is folded away: each row
    names its owner as (idx_type, offset) — the owner's EntityKind and its packed
    position within that type's block — and the model sum-pools rows back into
    their owner. The pending halt owns no token: idx_type == OWNER_PENDING,
    offset 0. Pads are idx_type == -1 (offset 0, x zeros)."""

    x: torch.Tensor  # (B, E, ENCODING_DIM_EFFECT) / E = the batch's longest row count
    x_entity_kind: torch.Tensor  # (B, E) long — owner `EntityKind` / `OWNER_PENDING` for the halt / -1 = pad
    x_offset: torch.Tensor  # (B, E) long / owner's packed position within its type's `X_Entity`

    @property
    def mask(self) -> torch.Tensor:
        return self.x_entity_kind >= 0


@tensorclass
class X_Modifiers(X_Owned):
    pass


@tensorclass
class X_Effects(X_Owned):
    pass


@tensorclass
class X_HealthBlock(X_Owned):
    pass


@tensorclass
class X_ShopPrices:
    """Price rows of the shop items that have a token, packed to a valid prefix in
    kind order (CARD_SHOP, RELIC_SHOP, POTION_SHOP); `idxs` is each row's kind
    (-1 = pad). Row k of a kind prices token k of that kind — rank within kind."""

    x: torch.Tensor  # (B, S, ENCODING_DIM_PRICE) — S = the batch's most priced items
    idxs: torch.Tensor  # (B, S) int16

    @property
    def mask(self) -> torch.Tensor:
        return self.idxs >= 0


@tensorclass
class X_GameState:
    # One X_Entity per type, in EntityKind order. The window's kind row is the cat
    # of their `idxs` then `room_idxs`, written wherever a consumer needs it
    cards: X_Entity
    relics: X_Entity
    potions: X_Entity
    monsters: X_Entity
    event_options: X_Entity
    character: X_Entity
    health_block: X_HealthBlock  # the combatants' shared health/block family, by kind

    effects: X_Effects
    modifiers: X_Modifiers
    energy: torch.Tensor
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    room_node_idx: torch.Tensor  # (B, MAP_WIDTH) long
    room_idxs: torch.Tensor  # (B, MAP_WIDTH) int16 — ROOMS at each reachable next-row column
    map_meta: torch.Tensor
    focus: torch.Tensor
    ascension: torch.Tensor  # (B, ENCODING_DIM_ASCENSION) thermometer, bit i = level > i
    reward_meta: torch.Tensor
    shop_meta: torch.Tensor
    event_meta: torch.Tensor
    shop_prices: X_ShopPrices  # the shop tokens' prices, by kind and rank


@dataclass
class X_CoreOutput:
    """`global_` and `entities` are the actor pass's, read by the policy heads.
    `global_value` is the value head's: the privileged pass's where the draw pile
    shows an order the actor can't see, the actor pass's elsewhere."""

    global_: torch.Tensor  # (B, dim_global)
    global_value: torch.Tensor  # (B, dim_global)
    entities: X_Entity  # the refined window: x (B, window, dim_entity), idxs (B, window) long


@tensorclass
class X_EntityMask:
    """The masks over one type's block, token for token with its `idxs` (the
    heads cat the blocks in X_GameState field order, exactly as the tokens)."""

    # L2 pick: legal AND in the action type's pool, deduped by card identity
    legal: torch.Tensor  # (B, NUM_ACTION_TYPES, W)
    # picks whose legal actions carry a monster target (cards / potions)
    needs_target: torch.Tensor  # (B, W)
    # legal L3 targets: the alive monsters. The engine offers every alive monster
    # to every targeted action, so the set does not depend on the L2 pick
    is_target: torch.Tensor  # (B, W)


@tensorclass
class X_Mask:
    mask_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) — L1

    # One X_EntityMask per block, in X_GameState field order (rooms last)
    cards: X_EntityMask
    relics: X_EntityMask
    potions: X_EntityMask
    monsters: X_EntityMask
    event_options: X_EntityMask
    character: X_EntityMask
    rooms: X_EntityMask


class Level(IntEnum):
    """Action-hierarchy levels — column indices into X_Action's (B, len(Level)) tensors."""

    ACTION_TYPE = 0
    L1 = 1
    L2 = 2


@tensorclass
class X_Action:
    """The sampled hierarchy per row, engine-facing: `kind` and `rank` name the L2
    pick as the rank-th token of its kind (the engine index; for a room, the rank
    among the reachable columns), `target` the L3 pick as the engine monster
    index. -1 where a level didn't apply."""

    action_type: torch.Tensor  # (B,) long
    kind: torch.Tensor  # (B,) long — SliceKind of the L2 pick
    rank: torch.Tensor  # (B,) long
    target: torch.Tensor  # (B,) long
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

    game_state: X_GameState  # (N, ...)
    mask_batch: X_Mask  # (N, ...) — row-sliced per minibatch (mask_batch[rows])
    action_type: torch.Tensor  # (N,) recorded L1 ActionType pick
    kind: torch.Tensor  # (N,) recorded L2 pick: the kind (-1 if none) ...
    rank: torch.Tensor  # (N,) ... and the rank within it — the engine index
    target: torch.Tensor  # (N,) recorded L3 pick: the engine monster index (-1 if none)
    log_probs_old: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, K)
    returns: torch.Tensor  # (N, K)
    advantages: torch.Tensor  # (N, 1), summed over streams, normalized


# members() is declaration order == int() order, which is what masks/utils/heads
# index by
ACTION_TYPE_BY_INT = ActionType.members()
_NUM_ENGINE_ACTION_TYPES = len(ACTION_TYPE_BY_INT)


# Halting candidate pool -> the kind its idx indexes
#
# The engine resolves every halt with one `EffectPendingResolve {idx}` over
# `pending.target.candidate_pool`. Each halt must score exactly its pool's kind
# and route to that class's key net, so the policy splits the one engine type
# into a virtual type per pool and folds/unfolds at the boundary
# (masks.build_masks in, utils.engine_action out). At most one is ever legal:
# the engine holds a single effect_pending
PENDING_POOL_SLICE: dict[CandidatePool, SliceKind] = {
    CandidatePool.CardPileHand: SliceKind.CARD_PILE_HAND,
    CandidatePool.CardPileDraw: SliceKind.CARD_PILE_DRAW,
    CandidatePool.CardPileDiscard: SliceKind.CARD_PILE_DISCARD,
    CandidatePool.CardPileExhaust: SliceKind.CARD_PILE_EXHAUST,
    CandidatePool.CardPileDeck: SliceKind.CARD_PILE_DECK,
    CandidatePool.CardPileDiscover: SliceKind.CARD_PILE_DISCOVER,
    CandidatePool.CardPileEventRoll: SliceKind.CARD_PILE_EVENT_ROLL,
    CandidatePool.RelicEventRoll: SliceKind.RELIC_EVENT_ROLL,
    CandidatePool.PotionEventRoll: SliceKind.POTION_EVENT_ROLL,
}
# Policy ints: engine ints 0..N-1 keep their values (so
# ACTION_TYPE_BY_INT.index(ActionType.X) still works), virtuals append
PENDING_POOL_INT: dict[CandidatePool, int] = {
    pool: _NUM_ENGINE_ACTION_TYPES + i for i, pool in enumerate(PENDING_POOL_SLICE)
}
PENDING_INTS: frozenset = frozenset(PENDING_POOL_INT.values())
NUM_ACTION_TYPES = _NUM_ENGINE_ACTION_TYPES + len(PENDING_POOL_SLICE)

# Each index-taking action -> the kinds its selection may land on, keyed by
# POLICY int (virtuals are not engine members). Terminal actions absent.
# A pick is a window position; its kind gives the pool, its rank within that
# kind gives the engine index. RewardTakeCard (idx_bundle, idx_card): the kind
# is the bundle, the rank is the card. RewardSingingBowl names a bundle via
# the bundle's first card. Only CardPlay/PotionUse have an L3 (monster target)
_REWARD_BUNDLES = (  # one kind per reward bundle (card.py owns their packing)
    SliceKind.CARD_REWARD_0,
    SliceKind.CARD_REWARD_1,
    SliceKind.CARD_REWARD_2,
    SliceKind.CARD_REWARD_3,
    SliceKind.CARD_REWARD_4,
)
_ENGINE_POOL: dict[ActionType, tuple[SliceKind, ...]] = {
    ActionType.CardPlay: (SliceKind.CARD_PILE_HAND,),
    ActionType.EventOptionSelect: (SliceKind.EVENT_OPTIONS,),
    ActionType.PotionDiscard: (SliceKind.POTION_OWNED,),
    ActionType.PotionUse: (SliceKind.POTION_OWNED,),
    ActionType.RewardSingingBowl: _REWARD_BUNDLES,
    ActionType.RewardTakeCard: _REWARD_BUNDLES,
    ActionType.RewardTakePotion: (SliceKind.POTION_REWARD,),
    ActionType.RewardTakeRelic: (SliceKind.RELIC_REWARD,),
    ActionType.RoomSelect: (SliceKind.ROOMS,),
    ActionType.ShopBuyCard: (SliceKind.CARD_SHOP,),
    ActionType.ShopBuyPotion: (SliceKind.POTION_SHOP,),
    ActionType.ShopBuyRelic: (SliceKind.RELIC_SHOP,),
}
ACTION_TYPE_POOL: dict[int, tuple[SliceKind, ...]] = {
    **{int(at): pools for at, pools in _ENGINE_POOL.items()},
    **{PENDING_POOL_INT[pool]: (kind,) for pool, kind in PENDING_POOL_SLICE.items()},
}

# Action-type groups the mask builder and decoder special-case
AT_REWARD_CARD = frozenset((int(ActionType.RewardTakeCard), int(ActionType.RewardSingingBowl)))
AT_MONSTER_TARGETED = frozenset((int(ActionType.CardPlay), int(ActionType.PotionUse)))
AT_SHOP = frozenset(
    (int(ActionType.ShopBuyCard), int(ActionType.ShopBuyRelic), int(ActionType.ShopBuyPotion))
)

# Drift alarm: the engine's index-taking set must equal our engine-keyed entries
# plus the one resolver we split into the virtuals
_ACTION_TYPE_ARITY: list = [ACTION_SPEC_REGISTRY[m].arity for m in ACTION_TYPE_BY_INT]
_ENGINE_INDEX_TAKING = {
    at for at in range(_NUM_ENGINE_ACTION_TYPES) if _ACTION_TYPE_ARITY[at] != (0, 0)
}
assert _ENGINE_INDEX_TAKING == {int(a) for a in _ENGINE_POOL} | {
    int(ActionType.EffectPendingResolve)
}, (
    "engine index-taking actions changed: _ENGINE_POOL + EffectPendingResolve "
    f"!= {sorted(_ENGINE_INDEX_TAKING)}"
)
