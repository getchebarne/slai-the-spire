from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import final

import torch
from slai import ActionType
from slai import Card
from slai import Character
from slai import EventOption
from slai import Monster
from slai import Potion
from slai import Relic
from slai import Room

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
from src.rl.constants import MAX_SIZE_CARD_MATCH_PAIRS
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


# In packing order: TOKEN_KINDS_OF_ENTITY_KIND's Entity kinds, one after another, each its
# token kinds in order
class TokenKind(IntEnum):
    CARD_PILE_HAND = 0
    CARD_PILE_DISCOVER = 1
    CARD_PILE_SELECTED = 2
    CARD_REWARD_0 = 3
    CARD_REWARD_1 = 4
    CARD_REWARD_2 = 5
    CARD_REWARD_3 = 6
    CARD_REWARD_4 = 7
    CARD_SHOP = 8
    CARD_PILE_EVENT_ROLL = 9
    CARD_MATCH_FLIPPED = 10
    CARD_MATCH_PAIRS = 11
    CARD_PILE_STASIS = 12
    CARD_PLAY_QUEUE = 13
    CARD_PILE_NIGHTMARE = 14
    CARD_PILE_DRAW = 15
    CARD_PILE_DISCARD = 16
    CARD_PILE_EXHAUST = 17
    CARD_PILE_DECK = 18
    RELIC_REWARD = 19
    RELIC_SHOP = 20
    RELIC_EVENT_ROLL = 21
    RELIC_OWNED = 22
    POTION_OWNED = 23
    POTION_REWARD = 24
    POTION_SHOP = 25
    POTION_EVENT_ROLL = 26
    MONSTERS = 27
    EVENT_OPTIONS = 28
    CHARACTER = 29
    ROOMS = 30


TOKEN_KINDS_CARD_REWARDS = (  # One token kind per Reward bundle (card.py owns their packing)
    TokenKind.CARD_REWARD_0,
    TokenKind.CARD_REWARD_1,
    TokenKind.CARD_REWARD_2,
    TokenKind.CARD_REWARD_3,
    TokenKind.CARD_REWARD_4,
)

# Pack order within the Card Entity kind: decision pools first, observational piles last
TOKEN_KINDS_CARD = (
    TokenKind.CARD_PILE_HAND,
    TokenKind.CARD_PILE_DISCOVER,
    TokenKind.CARD_PILE_SELECTED,
    *TOKEN_KINDS_CARD_REWARDS,
    TokenKind.CARD_SHOP,
    TokenKind.CARD_PILE_EVENT_ROLL,
    TokenKind.CARD_MATCH_FLIPPED,
    TokenKind.CARD_MATCH_PAIRS,
    TokenKind.CARD_PILE_STASIS,
    TokenKind.CARD_PLAY_QUEUE,
    TokenKind.CARD_PILE_NIGHTMARE,
    TokenKind.CARD_PILE_DRAW,
    TokenKind.CARD_PILE_DISCARD,
    TokenKind.CARD_PILE_EXHAUST,
    TokenKind.CARD_PILE_DECK,
)

# Pack order within the Relic Entity kind: decision pools first, the owned Relics last
TOKEN_KINDS_RELIC = (
    TokenKind.RELIC_REWARD,
    TokenKind.RELIC_SHOP,
    TokenKind.RELIC_EVENT_ROLL,
    TokenKind.RELIC_OWNED,
)

# Pack order within the Potion Entity kind; every source is engine-bounded
TOKEN_KINDS_POTION = (
    TokenKind.POTION_OWNED,
    TokenKind.POTION_REWARD,
    TokenKind.POTION_SHOP,
    TokenKind.POTION_EVENT_ROLL,
)

# The Entity kinds in window order, each its token kinds; the last four hold a single token kind
TOKEN_KINDS_OF_ENTITY_KIND = (
    TOKEN_KINDS_CARD,
    TOKEN_KINDS_RELIC,
    TOKEN_KINDS_POTION,
    (TokenKind.MONSTERS,),
    (TokenKind.EVENT_OPTIONS,),
    (TokenKind.CHARACTER,),
    (TokenKind.ROOMS,),
)

# Each token kind's token cap: the slice its encoder applies to that collection
MAP_TOKEN_KIND_CAP: dict[TokenKind, int] = {
    TokenKind.CARD_PILE_HAND: MAX_SIZE_CARD_PILE_HAND,
    TokenKind.CARD_PILE_DISCOVER: MAX_SIZE_CARD_PILE_DISCOVER,
    TokenKind.CARD_PILE_SELECTED: MAX_SIZE_CARD_PILE_SELECTED,
    TokenKind.CARD_REWARD_0: MAX_CARD_REWARD_ROLL,
    TokenKind.CARD_REWARD_1: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_2: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_3: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_4: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_SHOP: MAX_SIZE_SHOP_CARDS,
    TokenKind.CARD_PILE_EVENT_ROLL: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    TokenKind.CARD_MATCH_FLIPPED: 1,  # Match and Keep's face-up Card
    TokenKind.CARD_MATCH_PAIRS: MAX_SIZE_CARD_MATCH_PAIRS,
    TokenKind.CARD_PILE_STASIS: MAX_SIZE_CARD_PILE_STASIS,
    TokenKind.CARD_PLAY_QUEUE: MAX_SIZE_CARD_PLAY_QUEUE,
    TokenKind.CARD_PILE_NIGHTMARE: MAX_SIZE_CARD_PILE_NIGHTMARE,
    TokenKind.CARD_PILE_DRAW: MAX_SIZE_CARD_PILE_DRAW,
    TokenKind.CARD_PILE_DISCARD: MAX_SIZE_CARD_PILE_DISCARD,
    TokenKind.CARD_PILE_EXHAUST: MAX_SIZE_CARD_PILE_EXHAUST,
    TokenKind.CARD_PILE_DECK: MAX_SIZE_CARD_PILE_DECK,
    TokenKind.RELIC_REWARD: MAX_RELIC_REWARDS,
    TokenKind.RELIC_SHOP: MAX_SHOP_RELICS,
    TokenKind.RELIC_EVENT_ROLL: MAX_SIZE_RELIC_EVENT_ROLL,
    TokenKind.RELIC_OWNED: MAX_RELICS,
    TokenKind.POTION_OWNED: MAX_POTION_SLOTS,
    TokenKind.POTION_REWARD: MAX_POTION_REWARDS,
    TokenKind.POTION_SHOP: MAX_SHOP_POTIONS,
    TokenKind.POTION_EVENT_ROLL: MAX_SIZE_POTION_EVENT_ROLL,
    TokenKind.MONSTERS: MAX_MONSTERS,
    TokenKind.EVENT_OPTIONS: MAX_EVENT_OPTIONS,
    TokenKind.CHARACTER: 1,
    TokenKind.ROOMS: MAP_WIDTH,
}

# An Entity behind a token
type Entity = Card | Relic | Potion | Monster | EventOption | Character | Room


def make_channel_allocator() -> tuple[list[tuple[str, int]], Callable[..., int]]:
    """Make a (layout, alloc) pair.

    `alloc(channel, width=1)` appends and returns the offset, so offsets, widths and the total
    all derive from one list.

    Returns
    -------
    layout : list[tuple[str, int]]
        Every allocated channel and its width, in allocation order.
    alloc : Callable[..., int]
        The allocator over `layout`.
    """
    layout: list[tuple[str, int]] = []

    def alloc(channel: str, width: int = 1) -> int:
        offset = sum(w for _, w in layout)
        layout.append((channel, width))
        return offset

    return layout, alloc


class EffectSource(IntEnum):
    """What applies the Effect, and for Relics on which trigger.

    Stamped into the row by the Effects encoder, outside the row cache.
    """

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


class Tag(IntEnum):
    """An archetype a Card, Relic or Potion belongs to; one can belong to several."""

    POISON = 0
    DISCARD = 1
    NINJA = 2
    SETUP = 3  # Meant for planned combo plays


@final
@dataclass
class Jagged[T]:
    """Consecutive variable-length segments of a tensor or row record.

    `lengths[i]` is the number of logical rows in segment i. A tensor has one value per row on
    its leading axis. A Tokens record has one value per token in its direct tensors and one
    row or segment per token in its nested records. A Sparse record has one length per logical
    row, with columns and values on its entry axis. Only populated rows belong here; shared
    storage capacity is a separate concern.
    """

    lengths: torch.Tensor  # (number of segments,) integer, nonnegative
    rows: T


@final
@dataclass
class Sparse:
    """Sparse rows with consecutive nonzero entries, ordered by row then column.

    In a populated batch, lengths sum to the number of columns and values. Private worker
    storage may reserve extra capacity; read_slot exposes only its populated prefixes.
    """

    lengths: torch.Tensor  # (R,) int16, entries per row, including empty rows
    columns: torch.Tensor  # (F,) int16
    values: torch.Tensor  # (F,) float32


@final
@dataclass
class Tokens:
    """Every Entity token as a row, every state's in window order, state after state.

    Sparse features index the shared column space, each Entity kind's columns from its
    COL_START, then the health/block, price and slot columns Entity kinds share and the Tag
    columns Cards, Relics and Potions share. Effects and Modifiers each have one segment per
    token; the projector adds their rows into that token.
    """

    features: Sparse
    kind: torch.Tensor  # (N,) int16, which also names the Entity kind

    # Legal under the action type: an L2 selection, deduped by Card identity, or an L3 target
    legal: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Selections whose legal actions carry a Monster target (Cards / Potions)
    needs_target: torch.Tensor  # (N,) bool

    node_idx: torch.Tensor  # (N,) int16: a Room token's Map node; 0 on every other token

    # The rows the tokens own
    effects: Jagged[Sparse]  # Any number per token; lengths are int16
    modifiers: Jagged[Sparse]  # Any number per Character / Monster; int16


@dataclass
class StateBatch:
    """Compact game states held by the encoders, model, rollout and minibatches.

    Tokens and pending Effects each have one segment per state. Tokens in turn own segments
    of Effects and Modifiers. All other fields have one row per state. `src/rl/packed.py`
    selects and joins states using those explicit segment boundaries.
    """

    tokens: Jagged[Tokens]  # (B,) long lengths
    pending: Jagged[Sparse]  # (B,) long lengths, 0 or 1
    context: torch.Tensor  # (B, ENCODING_DIM_CONTEXT) float32; encoding.context owns its layout
    map_grid: torch.Tensor  # (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) float32
    legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1


@dataclass(frozen=True)
class MinibatchPlan:
    """Selected device state IDs and host integers sizing every selected table."""

    state_ids: torch.Tensor
    tokens: int
    effects: int
    modifiers: int
    pending: int
    token_entries: int
    effect_entries: int
    modifier_entries: int
    pending_entries: int


@dataclass(frozen=True)
class BatchLayout:
    """Cached boundaries for one immutable game batch; never a mutable worker slot.

    Device offsets index each Jagged's rows. CPU state boundaries also permit contiguous
    views without reading device scalars. Rebuild this layout if the batch's lengths change.
    """

    token_offsets: torch.Tensor  # (B + 1,) int64
    effect_offsets: torch.Tensor  # (N_tokens + 1,) int64
    modifier_offsets: torch.Tensor  # (N_tokens + 1,) int64
    pending_offsets: torch.Tensor  # (B + 1,) int64
    token_entry_offsets: torch.Tensor  # (N_tokens + 1,) int64
    effect_entry_offsets: torch.Tensor  # (N_effects + 1,) int64
    modifier_entry_offsets: torch.Tensor  # (N_modifiers + 1,) int64
    pending_entry_offsets: torch.Tensor  # (N_pending + 1,) int64
    state_offsets_cpu: torch.Tensor  # (B + 1, 8) int64, in MinibatchPlan's totals order


@dataclass
class TokenIndex:
    """Cached lookups for the model's token rows, built once by the projector.

    Token rows stay in the encoded batch's order through projection and attention, so these
    indices remain valid throughout the model pass. State boundaries belong to Jagged.lengths;
    action masks belong to StateBatch.tokens.rows. Host entity counts share the projector's
    one device read with the draw-order sizes.
    """

    kind: torch.Tensor  # (N,) long, token kind widened for indexing and engine-facing actions
    entity_kind: torch.Tensor  # (N,) long
    state: torch.Tensor  # (N,) long, the state owning each token
    kind_row: torch.Tensor  # (B, len(TokenKind)) long, N where the token kind is absent
    kind_count: torch.Tensor  # (B, len(TokenKind)) long
    entity_counts: list[int]  # One per Entity kind, in TOKEN_KINDS_OF_ENTITY_KIND's order


@dataclass
class CoreOutput:
    """Refined token embeddings, cached indices, and per-state policy/value contexts.

    The policy's tokens and context include draw order only where Frozen Eye makes it visible.
    The value context includes draw order on every state whose pile shows an order. Both
    contexts have one row per state; tokens have a variable number of rows per state.
    """

    tokens: Jagged[torch.Tensor]  # (B,) lengths; (N, dim_entity) refined token rows
    index: TokenIndex
    policy_context: torch.Tensor  # (B, dim_global)
    value_context: torch.Tensor  # (B, dim_global)


@dataclass
class PolicyOutput:
    """One sampled action per state, with the policy statistics for its three levels.

    Each action row holds [action type, token offset, target rank]. The token offset indexes
    that state's encoded tokens, across all token kinds; it stays valid when whole states are
    reordered or joined into a new batch. The target is a Monster's rank. A selection or target
    that did not apply holds -1. `actions.engine_action` resolves these coordinates to engine
    Entity ids using the encoder's token ordering.
    """

    actions: torch.Tensor  # (B, 3) long; action type, state-local token offset, Monster rank
    log_prob: torch.Tensor  # (B, 3) float, 0 where N/A
    entropy: torch.Tensor  # (B, 3) float, 0 where N/A
    kl_uniform: torch.Tensor  # (B, 3) float, KL(uniform‖π) over legal actions, 0 where N/A


@dataclass
class Rollout:
    """What the Actor collected, before GAE.

    T steps of E games, state t * E + e of the N = T * E states. `final_states` holds each
    game's state after the last step, for the bootstrap value.
    """

    states: StateBatch  # N states
    counts_cpu: torch.Tensor  # (N, number of blocks) long, in count_blocks' order
    actions: torch.Tensor  # (N, 3): action type, state-local token offset, Monster rank
    log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    rewards: torch.Tensor  # (T, E, K)
    dones: torch.Tensor  # (T, E)
    final_states: StateBatch  # E states


# members() is declaration order == int() order, which is what masks and heads index by
ACTION_TYPE_BY_INT = ActionType.members()

# The policy's action types are exactly the engine's
NUM_ACTION_TYPES = len(ACTION_TYPE_BY_INT)


def _check_tables() -> None:
    """Check that the tables that restate the engine or an enum still agree with it."""
    if len(TOKEN_KINDS_CARD_REWARDS) != MAX_REWARD_CARD_BUNDLES:
        raise RuntimeError(
            f"The engine stages up to {MAX_REWARD_CARD_BUNDLES} reward bundles: "
            "add CARD_REWARD_<b> token kinds"
        )
    if set(MAP_TOKEN_KIND_CAP) != set(TokenKind):
        raise RuntimeError(
            "A TokenKind has no entry in MAP_TOKEN_KIND_CAP: "
            f"{set(TokenKind) - set(MAP_TOKEN_KIND_CAP)}"
        )

    # TokenKind is declared in packing order, valued 0..len-1: the Entity kind tables, one after
    # another, list it, every token kind in one table
    token_kinds = [
        token_kind
        for token_kinds_entity_kind in TOKEN_KINDS_OF_ENTITY_KIND
        for token_kind in token_kinds_entity_kind
    ]
    if token_kinds != list(TokenKind) or token_kinds != list(range(len(token_kinds))):
        names = [token_kind.name for token_kind in token_kinds]
        raise RuntimeError(
            "Declare TokenKind in packing order, valued 0..len-1: TOKEN_KINDS_OF_ENTITY_KIND's "
            f"entity kinds, one after another, list {names}"
        )


_check_tables()
