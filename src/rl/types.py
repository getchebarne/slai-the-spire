from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from enum import StrEnum

import torch
from slai import ActionType
from slai import Effect
from tensordict import tensorclass

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_CARD_REWARD_ROLL
from src.rl.constants import MAX_COMBAT_CARD_REWARD
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_REWARD_CARD_BUNDLES
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_EXHAUST
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_CARD_PILE_NIGHTMARE
from src.rl.constants import MAX_SIZE_CARD_PILE_SELECTED
from src.rl.constants import MAX_SIZE_CARD_PILE_STASIS
from src.rl.constants import MAX_SIZE_CARD_PLAY_QUEUE
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.constants import MAX_SIZE_SHOP_CARDS


# Values index Core._type_emb rows, so checkpoints pin them: append new kinds, never renumber
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
    CARD_PILE_SELECTED = 26
    CARD_PLAY_QUEUE = 27
    CARD_PILE_NIGHTMARE = 28
    CARD_MATCH_FLIPPED = 29


SLICE_KINDS_CARD_REWARDS = (  # one kind per reward bundle (card.py owns their packing)
    SliceKind.CARD_REWARD_0,
    SliceKind.CARD_REWARD_1,
    SliceKind.CARD_REWARD_2,
    SliceKind.CARD_REWARD_3,
    SliceKind.CARD_REWARD_4,
)

# Pack order within the card block: decision pools first, observational piles last
SLICE_KINDS_CARD = (
    SliceKind.CARD_PILE_HAND,
    SliceKind.CARD_PILE_DISCOVER,
    SliceKind.CARD_PILE_SELECTED,
    *SLICE_KINDS_CARD_REWARDS,
    SliceKind.CARD_SHOP,
    SliceKind.CARD_PILE_EVENT_ROLL,
    SliceKind.CARD_MATCH_FLIPPED,
    SliceKind.CARD_PILE_STASIS,
    SliceKind.CARD_PLAY_QUEUE,
    SliceKind.CARD_PILE_NIGHTMARE,
    SliceKind.CARD_PILE_DRAW,
    SliceKind.CARD_PILE_DISCARD,
    SliceKind.CARD_PILE_EXHAUST,
    SliceKind.CARD_PILE_DECK,
)

# Pack order within the relic block: decision pools first, the owned relics last
SLICE_KINDS_RELIC = (
    SliceKind.RELIC_REWARD,
    SliceKind.RELIC_SHOP,
    SliceKind.RELIC_EVENT_ROLL,
    SliceKind.RELIC_OWNED,
)

# Pack order within the potion block; every source is engine-bounded
SLICE_KINDS_POTION = (
    SliceKind.POTION_OWNED,
    SliceKind.POTION_REWARD,
    SliceKind.POTION_SHOP,
    SliceKind.POTION_EVENT_ROLL,
)

# Each kind's token cap: the slice its encoder applies to that collection
CAP_OF_KIND: dict[SliceKind, int] = {
    SliceKind.CARD_PILE_HAND: MAX_SIZE_CARD_PILE_HAND,
    SliceKind.CARD_PILE_DISCOVER: MAX_SIZE_CARD_PILE_DISCOVER,
    SliceKind.CARD_PILE_SELECTED: MAX_SIZE_CARD_PILE_SELECTED,
    SliceKind.CARD_REWARD_0: MAX_CARD_REWARD_ROLL,
    SliceKind.CARD_REWARD_1: MAX_COMBAT_CARD_REWARD,
    SliceKind.CARD_REWARD_2: MAX_COMBAT_CARD_REWARD,
    SliceKind.CARD_REWARD_3: MAX_COMBAT_CARD_REWARD,
    SliceKind.CARD_REWARD_4: MAX_COMBAT_CARD_REWARD,
    SliceKind.CARD_SHOP: MAX_SIZE_SHOP_CARDS,
    SliceKind.CARD_PILE_EVENT_ROLL: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    SliceKind.CARD_MATCH_FLIPPED: 1,  # Match and Keep's face-up card
    SliceKind.CARD_PILE_STASIS: MAX_SIZE_CARD_PILE_STASIS,
    SliceKind.CARD_PLAY_QUEUE: MAX_SIZE_CARD_PLAY_QUEUE,
    SliceKind.CARD_PILE_NIGHTMARE: MAX_SIZE_CARD_PILE_NIGHTMARE,
    SliceKind.CARD_PILE_DRAW: MAX_SIZE_CARD_PILE_DRAW,
    SliceKind.CARD_PILE_DISCARD: MAX_SIZE_CARD_PILE_DISCARD,
    SliceKind.CARD_PILE_EXHAUST: MAX_SIZE_CARD_PILE_EXHAUST,
    SliceKind.CARD_PILE_DECK: MAX_SIZE_CARD_PILE_DECK,
    SliceKind.RELIC_REWARD: MAX_RELIC_REWARDS,
    SliceKind.RELIC_SHOP: MAX_SHOP_RELICS,
    SliceKind.RELIC_EVENT_ROLL: MAX_SIZE_RELIC_EVENT_ROLL,
    SliceKind.RELIC_OWNED: MAX_RELICS,
    SliceKind.POTION_OWNED: MAX_POTION_SLOTS,
    SliceKind.POTION_REWARD: MAX_POTION_REWARDS,
    SliceKind.POTION_SHOP: MAX_SHOP_POTIONS,
    SliceKind.POTION_EVENT_ROLL: MAX_SIZE_POTION_EVENT_ROLL,
    SliceKind.MONSTERS: MAX_MONSTERS,
    SliceKind.EVENT_OPTIONS: MAX_EVENT_OPTIONS,
    SliceKind.CHARACTER: 1,
    SliceKind.ROOMS: MAP_WIDTH,
}


def make_channel_allocator() -> tuple[list[tuple[StrEnum, int]], Callable[..., int]]:
    """A (layout, alloc) pair: alloc(channel, width=1) appends and returns the
    offset, so offsets, widths and the total all derive from one list."""
    layout: list[tuple[StrEnum, int]] = []

    def alloc(channel: StrEnum, width: int = 1) -> int:
        offset = sum(w for _, w in layout)
        layout.append((channel, width))
        return offset

    return layout, alloc


class EntityKind(IntEnum):
    """The one entity-type order: X_GameState's entity fields, the model's block
    cat, and the owner type of an X_Owned row all follow it."""

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


# An effect before encoding: the offset of the token that owns it (X_Effects.x_offset),
# what applies it, and the effect
type EffectRow = tuple[int, EffectSource, Effect]

# One list of EffectRows per state in the batch
type BatchEffectRows = list[list[EffectRow]]

# A shop token's price before encoding: the offset of the token that sells it, and its price
type PriceRow = tuple[int, int]

# One list of PriceRows per state in the batch
type BatchPriceRows = list[list[PriceRow]]


@tensorclass
class X_Entity:
    """Dense tokens and their kinds: the map GNN's rooms, and the projector's window (every
    type's block, then the rooms); `x_slice_kind[b, w]` is token w's SliceKind, -1 on a pad."""

    x: torch.Tensor  # (B, W, D_type)
    x_slice_kind: torch.Tensor  # (B, W) int16 (long once the model owns the window)

    @property
    def mask(self) -> torch.Tensor:
        return self.x_slice_kind >= 0


@tensorclass
class X_EntityPairs:
    """One entity type's encoded tokens, packed to a valid prefix; `x_slice_kind[b, w]` is
    token w's SliceKind, -1 past the prefix (pad). W is the batch's longest packed
    sequence of the type, and K its most nonzero columns in a row — no fixed width anywhere
    but X_Rows' caps. Each row is its nonzero (column, value) pairs in column order, then
    (0, 0.0) pairs up to K."""

    x_column: torch.Tensor  # (B, W, K) int16
    x_value: torch.Tensor  # (B, W, K) float32
    x_slice_kind: torch.Tensor  # (B, W) int16

    @property
    def mask(self) -> torch.Tensor:
        return self.x_slice_kind >= 0


@tensorclass
class X_Owned:
    """Rows owned by tokens, one family per subclass, which the projector adds into
    their owners. Pads are x_entity_kind == -1, with zeros elsewhere."""

    x_column: torch.Tensor  # (B, R, K) int16 — as in X_EntityPairs; R = the batch's most rows
    x_value: torch.Tensor  # (B, R, K) float32
    x_entity_kind: torch.Tensor  # (B, R) long — owner `EntityKind` / -1 = pad
    x_offset: torch.Tensor  # (B, R) long — owner's packed position in its type's `X_EntityPairs`

    @property
    def mask(self) -> torch.Tensor:
        return self.x_entity_kind >= 0


@tensorclass
class X_Modifiers(X_Owned):
    """Any number per Character / Monster."""


@tensorclass
class X_Effects(X_Owned):
    """Any number per token; the pending halt owns none (OWNER_PENDING, offset 0)."""


@tensorclass
class X_HealthBlock(X_Owned):
    """One per Character / Monster."""


@tensorclass
class X_Prices(X_Owned):
    """One per Shop token (Card / Relic / Potion)."""


@tensorclass
class X_GameState:
    # One X_EntityPairs per type, in EntityKind order. The window's kind row is the cat
    # of their `x_slice_kind` then `room_slice_kind`, written wherever a consumer needs it
    cards: X_EntityPairs
    relics: X_EntityPairs
    potions: X_EntityPairs
    monsters: X_EntityPairs
    event_options: X_EntityPairs
    character: X_EntityPairs
    health_block: X_HealthBlock  # the combatants' shared health/block family, by kind
    prices: X_Prices  # the shop tokens' shared price family, by owner

    effects: X_Effects
    modifiers: X_Modifiers
    energy: torch.Tensor  # (B, ENCODING_DIM_ENERGY) float32
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    room_node_idx: torch.Tensor  # (B, MAP_WIDTH) long
    room_slice_kind: torch.Tensor  # (B, MAP_WIDTH) int16 — ROOMS at each reachable next-row column
    map_meta: torch.Tensor  # (B, ENCODING_DIM_MAP_META) float32
    focus: torch.Tensor  # (B, ENCODING_DIM_FOCUS) float32
    ascension: torch.Tensor  # (B, ENCODING_DIM_ASCENSION) thermometer, bit i = level > i
    reward_meta: torch.Tensor  # (B, ENCODING_DIM_REWARD_META) float32
    shop_meta: torch.Tensor  # (B, ENCODING_DIM_SHOP_META) float32
    event_meta: torch.Tensor  # (B, ENCODING_DIM_EVENT_META) float32


@dataclass
class X_CoreOutput:
    """`global_` and `entities` are the policy heads': the transformer's, but the draw-order
    layer's on the rows under Frozen Eye whose draw pile shows an order. `global_value` is the
    value head's: the layer's on every row whose draw pile shows an order, the transformer's
    elsewhere."""

    global_: torch.Tensor  # (B, dim_global)
    global_value: torch.Tensor  # (B, dim_global)
    entities: X_Entity  # refined window: x (B, window, dim_entity), x_slice_kind (B, window) long


@tensorclass
class X_EntityMask:
    """The masks over one type's block, token for token with its `x_slice_kind` (the
    heads cat the blocks in X_GameState field order, exactly as the tokens)."""

    # L2 selection: legal under the action type, deduped by card identity
    legal: torch.Tensor  # (B, NUM_ACTION_TYPES, W)

    # selections whose legal actions carry a monster target (cards / potions)
    needs_target: torch.Tensor  # (B, W)


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

    # legal L3 targets: the alive monsters. The engine offers every alive monster
    # to every targeted action, so the set does not depend on the L2 selection
    monster_is_target: torch.Tensor  # (B, W) — W = the monster block's width


@tensorclass
class X_Action:
    """The sampled hierarchy per row, engine-facing: `kind` and `rank` name the L2
    selection as the rank-th token of its kind, `target` the L3 selection as the rank-th monster
    token, and masks.engine_action names their entities by id. -1 where a level didn't
    apply."""

    action_type: torch.Tensor  # (B,) long
    kind: torch.Tensor  # (B,) long — SliceKind of the L2 selection
    rank: torch.Tensor  # (B,) long
    target: torch.Tensor  # (B,) long
    log_prob: torch.Tensor  # (B, 3) float, 0 where N/A
    entropy: torch.Tensor  # (B, 3) float, 0 where N/A
    kl_uniform: torch.Tensor  # (B, 3) float, KL(uniform‖π) over legal actions, 0 where N/A


@tensorclass
class X_Rows:
    """One row per game, shared by the step loop and its workers: the action the step loop
    sampled, then the reward, done flag, state and masks the worker wrote back"""

    actions: torch.Tensor  # (E, 4) long: action type, kind, rank, target
    rewards: torch.Tensor  # (E, K)
    dones: torch.Tensor  # (E,)
    game_state: X_GameState  # (E, ...), every block at its cap
    mask_batch: X_Mask  # (E, ...), every block at its cap


@tensorclass
class Rollout:
    """What the actor collected, before GAE: T steps of E games, row t * E + e of the
    N = T * E rows. The `final_` field is each game's state after the last step, for
    the bootstrap value"""

    game_state: X_GameState  # (N, ...)
    mask_batch: X_Mask  # (N, ...)
    actions: torch.Tensor  # (N, 4): action type, kind, rank, target
    log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    rewards: torch.Tensor  # (T, E, K)
    dones: torch.Tensor  # (T, E)
    final_game_state: X_GameState  # (E, ...)


@tensorclass
class TrainingBatch:
    """Columnar rollout (N = rollout_length * num_envs rows). The full-batch masks are
    stored once and row-sliced per minibatch; GAE returns/advantages are precomputed.
    Values/returns are per reward stream (K = len(RewardStream)); the policy
    advantage is the per-stream advantages summed, then normalized."""

    game_state: X_GameState  # (N, ...)
    mask_batch: X_Mask  # (N, ...) — row-sliced per minibatch (mask_batch[rows])
    actions: torch.Tensor  # (N, 4): action type, kind, rank, target
    log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    log_probs_prox: torch.Tensor  # (N,) the PPO anchor: the clip and the KL measure from it
    values_prox: torch.Tensor  # (N, K) the values that go with π_prox
    returns: torch.Tensor  # (N, K)
    advantages: torch.Tensor  # (N, 1), summed over streams, normalized


# members() is declaration order == int() order, which is what masks and heads index by
ACTION_TYPE_BY_INT = ActionType.members()

# The policy's action types are exactly the engine's
NUM_ACTION_TYPES = len(ACTION_TYPE_BY_INT)


def _check_tables() -> None:
    """The tables that restate the engine or an enum still agree with it"""
    if len(SLICE_KINDS_CARD_REWARDS) != MAX_REWARD_CARD_BUNDLES:
        raise RuntimeError(
            f"The engine stages up to {MAX_REWARD_CARD_BUNDLES} reward bundles: "
            "add CARD_REWARD_<b> kinds"
        )
    if set(CAP_OF_KIND) != set(SliceKind):
        raise RuntimeError(
            f"A SliceKind has no entry in CAP_OF_KIND: {set(SliceKind) - set(CAP_OF_KIND)}"
        )


_check_tables()
