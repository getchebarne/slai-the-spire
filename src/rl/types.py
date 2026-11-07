from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import final

import torch
from slai import Card
from slai import Character
from slai import EventOption
from slai import Monster
from slai import Potion
from slai import Relic
from slai import Room


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


# An Entity behind a token
type Entity = Card | Relic | Potion | Monster | EventOption | Character | Room
type Token = tuple[TokenKind, Entity]


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


class ModifierOwner(IntEnum):
    """The combatant carrying the Modifier."""

    CHARACTER = 0
    MONSTER = 1


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

    `x_lengths[i]` is the number of logical rows in segment i. A tensor has one value per row on
    its leading axis. A Tokens record has one value per token in its direct tensors and one
    row or segment per token in its nested records. A Sparse record has one length per logical
    row, with columns and values on its entry axis. Only populated rows belong here; shared
    storage capacity is a separate concern.
    """

    x_lengths: torch.Tensor  # (number of segments,) integer, nonnegative
    rows: T


@final
@dataclass
class Sparse:
    """Sparse rows with consecutive nonzero entries, ordered by row then column.

    In a populated batch, x_lengths sum to the number of x_columns and x_values. Private worker
    storage may reserve extra capacity; read_slot exposes only its populated prefixes.
    """

    x_lengths: torch.Tensor  # (R,) int16, entries per row, including empty rows
    x_columns: torch.Tensor  # (F,) int16
    x_values: torch.Tensor  # (F,) float32


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
    x_kind: torch.Tensor  # (N,) int16, which also names the Entity kind

    # Items selectable under each action type, deduped by Card identity
    x_legal_selection: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Monster targets legal under each action type
    x_legal_target: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Selections whose legal actions carry a Monster target (Cards / Potions)
    x_needs_target: torch.Tensor  # (N,) bool

    x_node_idx: torch.Tensor  # (N,) int16: a Room token's Map node; 0 on every other token

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
    x_context: torch.Tensor  # (B, ENCODING_DIM_CONTEXT) float32; encoding.context owns its layout
    x_map_grid: torch.Tensor  # (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) float32
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1


@dataclass(frozen=True)
class MinibatchPlan:
    """Selected device state IDs and host integers sizing every selected table."""

    x_state_ids: torch.Tensor
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

    x_token_offsets: torch.Tensor  # (B + 1,) int64
    x_effect_offsets: torch.Tensor  # (N_tokens + 1,) int64
    x_modifier_offsets: torch.Tensor  # (N_tokens + 1,) int64
    x_pending_offsets: torch.Tensor  # (B + 1,) int64
    x_token_entry_offsets: torch.Tensor  # (N_tokens + 1,) int64
    x_effect_entry_offsets: torch.Tensor  # (N_effects + 1,) int64
    x_modifier_entry_offsets: torch.Tensor  # (N_modifiers + 1,) int64
    x_pending_entry_offsets: torch.Tensor  # (N_pending + 1,) int64
    x_state_offsets_cpu: torch.Tensor  # (B + 1, 8) int64, in MinibatchPlan's totals order


@dataclass
class TokenIndex:
    """Cached lookups for the model's token rows, built once by the projector.

    Token rows stay in the encoded batch's order through projection and attention, so these
    indices remain valid throughout the model pass. State boundaries belong to Jagged.x_lengths;
    action masks belong to StateBatch.tokens.rows. Host entity counts share the projector's
    one device read with the draw-order sizes.
    """

    x_kind: torch.Tensor  # (N,) long, Tokens.x_kind widened once for the projector and Core
    x_entity_kind: torch.Tensor  # (N,) long
    x_state: torch.Tensor  # (N,) long, the state owning each token
    x_kind_row: torch.Tensor  # (B, len(TokenKind)) long, N where the token kind is absent
    x_kind_count: torch.Tensor  # (B, len(TokenKind)) long
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
    x_policy_context: torch.Tensor  # (B, dim_global)
    x_value_context: torch.Tensor  # (B, dim_global)


@dataclass
class PolicyOutput:
    """One sampled action per state, with the policy statistics for its three levels.

    Each action row holds [action type, token offset, target rank]. The token offset indexes
    that state's encoded tokens, across all token kinds; it stays valid when whole states are
    reordered or joined into a new batch. The target is a Monster's rank. A selection or target
    that did not apply holds -1. `actions.engine_action` resolves these coordinates to engine
    Entity ids using the encoder's token ordering.
    """

    x_actions: torch.Tensor  # (B, 3) long; action type, state-local token offset, Monster rank
    x_log_prob: torch.Tensor  # (B, 3) float, 0 where N/A
    x_entropy: torch.Tensor  # (B, 3) float, 0 where N/A
    x_kl_uniform: torch.Tensor  # (B, 3) float, KL(uniform‖π) over legal actions, 0 where N/A


@dataclass
class Rollout:
    """What the Actor collected, before GAE.

    T steps of E games, state t * E + e of the N = T * E states. `final_states` holds each
    game's state after the last step, for the bootstrap value.
    """

    states: StateBatch  # N states
    x_counts_cpu: torch.Tensor  # (N, number of blocks) long, in count_blocks' order
    x_actions: torch.Tensor  # (N, 3): action type, state-local token offset, Monster rank
    x_log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    x_rewards: torch.Tensor  # (T, E, K)
    x_dones: torch.Tensor  # (T, E)
    final_states: StateBatch  # E states
