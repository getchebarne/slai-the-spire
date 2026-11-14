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


class EntityKind(IntEnum):
    """The kind of Entity behind a token."""

    CARD = 0
    CHARACTER = 1
    EVENT_OPTION = 2
    MONSTER = 3
    POTION = 4
    RELIC = 5
    ROOM = 6


class TokenKind(IntEnum):
    CARD_MATCH_FLIPPED = 0
    CARD_MATCH_PAIRS = 1
    CARD_PILE_DECK = 2
    CARD_PILE_DISCARD = 3
    CARD_PILE_DISCOVER = 4
    CARD_PILE_DRAW = 5
    CARD_PILE_EVENT_ROLL = 6
    CARD_PILE_EXHAUST = 7
    CARD_PILE_HAND = 8
    CARD_PILE_NIGHTMARE = 9
    CARD_PILE_SELECTED = 10
    CARD_PILE_STASIS = 11
    CARD_PLAY_QUEUE = 12
    CARD_REWARD_0 = 13
    CARD_REWARD_1 = 14
    CARD_REWARD_2 = 15
    CARD_REWARD_3 = 16
    CARD_REWARD_4 = 17
    CARD_SHOP = 18
    CHARACTER = 19
    EVENT_OPTIONS = 20
    MONSTERS = 21
    POTION_EVENT_ROLL = 22
    POTION_OWNED = 23
    POTION_REWARD = 24
    POTION_SHOP = 25
    RELIC_EVENT_ROLL = 26
    RELIC_OWNED = 27
    RELIC_REWARD = 28
    RELIC_SHOP = 29
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
    x_entity_kind: torch.Tensor  # (N,) int16, its token kind's Entity kind

    # Items selectable under each action type, deduped by Card identity
    x_legal_selection: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Monster targets legal under each action type
    x_legal_target: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Selections whose legal actions carry a Monster target (Cards / Potions)
    x_needs_target: torch.Tensor  # (N,) bool

    # The rows the tokens own
    effects: TOwned  # Any number per token; lengths are int16
    modifiers: TOwned  # Any number per Character / Monster; int16


@final
@dataclass
class StateCounts:
    """How much each state holds: a batch's shape.

    Every size is a sum over these counts and every position a running total. A batch on a
    device carries one copy; the same counts on the host travel beside it, so no size is ever
    read back from the device.
    """

    x_token_kind_count: torch.Tensor  # (B, len(TokenKind)) int16, its tokens of each token kind
    x_jagged_count: torch.Tensor  # (B, number_of_Jaggeds) int32, its rows in each, in Jagged order


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
    state_counts: StateCounts


# Sparse rows: x_lengths are (R,) int16, entries per row, including empty rows, and sum to F
type Sparse = Jagged[SparseEntries]

# The same rows in a worker's shared memory, at worst-case capacity
type SparseReserved = JaggedReserved[SparseEntries]

# Compact game states held by the model, rollout and minibatches
type StateBatch = States[Jagged[Tokens[Sparse, Jagged[Sparse]]], Sparse]

# The same states in a worker's shared memory, at worst-case capacity
type StateBatchReserved = States[
    JaggedReserved[Tokens[SparseReserved, JaggedReserved[SparseReserved]]], SparseReserved
]


@final
@dataclass
class TokensRefined:
    """Every Entity token after the core, as a row, in the input tokens' order.

    The embedding is the core's, and so is each token's state, derived once from the lengths.
    The Entity kind and the legal masks are the input tokens' own, carried over untouched so they
    stay with the tokens they describe.
    """

    x_embedding: torch.Tensor  # (N, dim_entity)
    x_token_state: torch.Tensor  # (N,) long, its state
    x_entity_kind: torch.Tensor  # (N,) int16, its token kind's Entity kind

    # Items selectable under each action type, deduped by Card identity
    x_legal_selection: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Monster targets legal under each action type
    x_legal_target: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # Selections whose legal actions carry a Monster target (Cards / Potions)
    x_needs_target: torch.Tensor  # (N,) bool


@dataclass
class CoreOutput:
    """The game states after the core: what the heads read.

    Tokens have one segment per state; every other field has one row per state. The legal action
    types and the counts are the input states' own, carried over untouched.
    """

    tokens: Jagged[TokensRefined]  # (B,) long lengths
    x_global: torch.Tensor  # (B, dim_global)
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1
    state_counts: StateCounts


@dataclass
class PolicyOutput:
    """One action per state, selected or recorded, with the policy statistics for its three levels.

    Each action row holds [action type, selection offset, target rank]. The selection offset
    indexes that state's encoded tokens, across all token kinds; it stays valid when whole states
    are reordered or joined into a new batch. The target is a Monster's rank. A selection or target
    that did not apply holds -1. `actions.engine_action` resolves these coordinates to engine
    Entity ids using the encoder's token ordering.
    """

    x_actions: torch.Tensor  # (B, 3) long; action type, state-local selection offset, Monster rank
    x_log_prob: torch.Tensor  # (B, 3) float, 0 where N/A
    x_entropy: torch.Tensor  # (B, 3) float, 0 where N/A
    x_kl_uniform: torch.Tensor  # (B, 3) float, KL-Uniform over legal actions, 0 where N/A


@dataclass
class Rollout:
    """What the Actor collected, before GAE.

    T steps of E games, state t * E + e of the B = T * E states. `state_batch_final` holds each
    game's state after the last step, for the bootstrap value.
    """

    state_batch: StateBatch  # B states
    x_actions: torch.Tensor  # (B, 3): action type, state-local selection offset, Monster rank
    x_log_probs_behavior: torch.Tensor  # (B,) under the weights that played
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
