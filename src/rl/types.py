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
class JaggedReserved[T]:
    """A Jagged's arrays at worst-case capacity, in a worker's shared memory.

    The encoders write a prefix of each array and the rows past it are stale, so nothing
    consumes it until `read_states` has generated the Jagged of its written rows.
    """

    x_lengths: torch.Tensor  # (segment capacity,) integer, nonnegative
    rows: T


@final
@dataclass
class SparseEntries:
    """The nonzero entries of sparse rows, consecutive and ordered by row then column."""

    x_columns: torch.Tensor  # (F,) int16
    x_values: torch.Tensor  # (F,) float32


# Sparse rows: x_lengths are (R,) int16, entries per row, including empty rows, and sum to F
type Sparse = Jagged[SparseEntries]

# The same rows in a worker's shared memory, at worst-case capacity
type SparseReserved = JaggedReserved[SparseEntries]


@final
@dataclass
class Tokens[TFeatures, TOwned]:
    """Every Entity token as a row, every state's in window order, state after state.

    Sparse features index the shared column space, each Entity kind's columns from its
    COL_START, then the health/block, price and slot columns Entity kinds share and the Tag
    columns Cards, Relics and Potions share. Effects and Modifiers each have one segment per
    token; the projector adds their rows into that token.
    """

    features: TFeatures
    x_token_kind: torch.Tensor  # (N,) int16, which also names the Entity kind

    # Items selectable under each action type, deduped by Card identity
    x_legal_selection: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Monster targets legal under each action type
    x_legal_target: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Selections whose legal actions carry a Monster target (Cards / Potions)
    x_needs_target: torch.Tensor  # (N,) bool

    x_node_idx: torch.Tensor  # (N,) int16: a Room token's Map node; 0 on every other token

    # The rows the tokens own
    effects: TOwned  # Any number per token; lengths are int16
    modifiers: TOwned  # Any number per Character / Monster; int16


@dataclass
class States[TTokens, TEffectPending]:
    """Game states in either form: finished (`StateBatch`) or worker memory (`StateBatchReserved`).

    Tokens have one segment per state, and each token owns segments of Effects and Modifiers.
    All other fields, the pending Effect included, have one row per state. `src/rl/batches.py`
    selects and joins states using those explicit segment boundaries. Jagged order is this
    record's field order, depth first.
    """

    tokens: TTokens  # (B,) long lengths
    effect_pending: TEffectPending  # (B,) int16 lengths, 0 where no halt is pending
    x_context: torch.Tensor  # (B, ENCODING_DIM_CONTEXT) float32; encoding.context owns its layout
    x_map_grid: torch.Tensor  # (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) uint8
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1


# Compact game states held by the model, rollout and minibatches
type StateBatch = States[Jagged[Tokens[Sparse, Jagged[Sparse]]], Sparse]

# The same states in a worker's shared memory, at worst-case capacity
type StateBatchReserved = States[
    JaggedReserved[Tokens[SparseReserved, JaggedReserved[SparseReserved]]], SparseReserved
]


@dataclass(frozen=True)
class MinibatchPlan:
    """Selected device state IDs and host integers sizing every Jagged's selected rows."""

    x_state_ids: torch.Tensor
    counts_total: list[int]  # One per Jagged, in Jagged order


@dataclass(frozen=True)
class StateBatchLayout:
    """Cached offsets for one immutable game batch; never a batch read from a worker's buffer.

    Device offsets index each Jagged's rows. CPU offsets also permit contiguous views
    without reading device scalars. Rebuild this layout if the batch's lengths change.
    """

    x_off_state: torch.Tensor  # (B + 1, number_of_Jaggeds) long, in Jagged order
    x_off_state_cpu: torch.Tensor  # (B + 1, number_of_Jaggeds) long, in Jagged order


@dataclass
class TokenIndex:
    """Cached lookups for the model's token rows, built once by the projector.

    Token rows stay in the encoded batch's order through projection and attention, so these
    indices remain valid throughout the model pass. State boundaries belong to Jagged.x_lengths;
    action masks belong to StateBatch.tokens.rows. Host entity counts share the projector's
    one device read with the draw-order sizes.
    """

    x_kind: torch.Tensor  # (N,) long, Tokens.x_token_kind widened once for the projector and Core
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

    T steps of E games, state t * E + e of the N = T * E states. `state_batch_final` holds each
    game's state after the last step, for the bootstrap value.
    """

    state_batch: StateBatch  # N states
    x_counts_cpu: torch.Tensor  # (N, number_of_Jaggeds) long, in Jagged order
    x_actions: torch.Tensor  # (N, 3): action type, state-local token offset, Monster rank
    x_log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    x_rewards: torch.Tensor  # (T, E, K)
    x_dones: torch.Tensor  # (T, E)
    state_batch_final: StateBatch  # E states


@dataclass
class RunState:
    """What a checkpoint resumes besides the weights, the optimizer and the iteration."""

    total_steps: int  # Steps played so far, every rank's games counted
    ascension_current: float  # The soft-curriculum position
    streak: int  # The consecutive passes at ascension_current so far
    lr_scale: float  # The Adaptive-KL learning rate scale
