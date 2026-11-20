from dataclasses import dataclass
from enum import IntEnum
from typing import Literal
from typing import final

import numpy as np
import torch
from slai import Card
from slai import Character
from slai import EventOption
from slai import Monster
from slai import PlaySource
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


class CollectionKind(IntEnum):
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

# The collection kinds of the collections holding an Entity, in window order, then the Entity
type Token = tuple[tuple[CollectionKind, ...], Entity]

# The record key of a Card, of a Relic and of a Potion: everything its token's row reads
type RecordKeyCard = tuple[int, int, int | None, int | None, int | None, PlaySource | None]
type RecordKeyRelic = tuple[int, bool, bool, int | None]
type RecordKeyPotion = tuple[int, bool, int | None]


class EffectSource(IntEnum):
    """What applies the Effect, and for Relics on which trigger.

    Stamped into the row by the Effects encoder.
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


# Collection storage forms, used only in type annotations
type Live = Literal["live"]
type Reserved = Literal["reserved"]


@final
@dataclass
class Sparse[TStorage]:
    """Nonzero feature entries, consecutive and ordered by row then column.

    `x_count` counts entries per feature row, including empty rows. The columns and values
    describe the same entries. Live collections hold exactly their populated rows and entries;
    Reserved collections hold capacity whose unused tails are stale. TStorage records this
    distinction in annotations without adding a runtime field.
    """

    x_col: torch.Tensor    # (F,) int16
    x_value: torch.Tensor  # (F,) float32
    x_count: torch.Tensor  # (R,) int16, entries per row


@final
@dataclass
class Dense:
    """One dense feature vector per token, in the tokens' order."""

    x_value: torch.Tensor  # (N, dim_entity)


@final
@dataclass
class OwnedRows[TStorage]:
    """Effect or Modifier rows, consecutive in their owning tokens' order.

    `x_count` counts rows per token. Each owned row has one entry count in `sparse`, even when
    that row has no nonzero features.
    """

    sparse: Sparse[TStorage]
    x_count: torch.Tensor  # (N,) int16, rows per token


@final
@dataclass
class EncodedFeatures[TStorage]:
    """The encoder's token features and the rows those tokens own.

    Token entries index the token columns, each Entity kind's columns from its COL_START,
    then the health/block, price and slot columns Entity kinds share and the Tag columns Cards,
    Relics and Potions share. The projector embeds the owned rows and adds them into each token.
    """

    sparse: Sparse[TStorage]  # One feature row per token

    # The rows the tokens own
    effects: OwnedRows[TStorage]    # Any number per token; counts are int16
    modifiers: OwnedRows[TStorage]  # Any number per Character / Monster; int16


@final
@dataclass
class Tokens[TFeatures]:
    """One token per Entity, each state's in window order, state after state.

    A token sits at its Entity's first collection in window order, and its collection kinds mark
    every collection holding the Entity. `x_count` counts tokens per state. Every token has
    features, kinds and legal masks. Features are EncodedFeatures before projection and Dense
    afterwards; projection and attention keep the token order and state boundaries.
    """

    features: TFeatures
    x_count: torch.Tensor  # (B,) long, tokens per state

    # Each token's `CollectionKind`s and `EntityKind`
    x_collection_kind: torch.Tensor  # (N_tok, len(CollectionKind)) uint8, multi-hot
    x_entity_kind: torch.Tensor      # (N_tok,) int16

    # Whether the token is selectable under the `ActionType`, deduped by Card identity
    x_legal_selection: torch.Tensor  # (N_tok,) long, a bit per `ActionType`

    # Whether the token is targetable by the `ActionType` (`CardPlay` / `PotionUse`)
    x_legal_target: torch.Tensor  # (N_tok,) long, a bit per `ActionType`

    # Whether the token needs an additional Monster target if selected (Cards / Potions)
    x_needs_target: torch.Tensor  # (N_tok,) bool


class StateCollection(IntEnum):
    """Columns of States.x_total after its per-collection-kind counts.

    These identities name collections explicitly; record field order does not determine them.
    """

    TOKENS = len(CollectionKind)
    TOKEN_ENTRIES = TOKENS + 1
    EFFECTS = TOKENS + 2
    EFFECT_ENTRIES = TOKENS + 3
    MODIFIERS = TOKENS + 4
    MODIFIER_ENTRIES = TOKENS + 5
    PENDING_ENTRIES = TOKENS + 6


@dataclass
class States[TStorage]:
    """Encoded game states, with explicit collections of tokens, owned rows and sparse entries.

    Tokens belong to states; Effects and Modifiers belong to tokens. The pending Effect has one
    sparse feature row per state, empty where no halt is pending. TStorage propagates Live or
    Reserved to these collections. `read_states` turns reserved capacity into populated views;
    rewriting that capacity invalidates its earlier views. The worker protocol owns access.

    `x_total` holds each state's derived totals on the data's device, and `x_total_cpu` holds
    the same totals on the CPU so output sizes need no device readback. CPU states use the same
    tensor for both fields. Batch operations keep them together. CollectionKind indexes the first
    columns, the tokens each collection holds, and StateCollection the rest.
    """

    tokens: Tokens[EncodedFeatures[TStorage]]
    effect_pending: Sparse[TStorage]

    # One row per state
    x_context: torch.Tensor            # (B, ENCODING_DIM_CONTEXT) float32; encoding.context owns its layout
    x_map_grid: torch.Tensor           # (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) uint8
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1
    x_total: torch.Tensor              # (B, len(CollectionKind) + len(StateCollection)) int32
    x_total_cpu: torch.Tensor          # Same totals on CPU; aliases x_total for CPU states


@dataclass
class CoreOutput:
    """The game states after the core: what the heads read.

    Dense token features preserve the input tokens' order, kinds and legal masks. The core also
    carries each token's state, derived once from the counts. Global vectors, legal action types
    and totals each have one row per state.
    """

    tokens: Tokens[Dense]
    x_token_state: torch.Tensor        # (N,) long, each token's state
    x_global: torch.Tensor             # (B, dim_global)
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1
    x_total: torch.Tensor              # (B, len(CollectionKind) + len(StateCollection)) int32


@dataclass
class PolicyOutput:
    """One action per state, selected or recorded, with the policy statistics for its three levels.

    Each action row holds [action type, selection offset, target rank]. The selection offset
    indexes that state's encoded tokens, one per Entity across all collection kinds; it stays valid
    when whole states are reordered or joined into a new batch. The target is a Monster's rank.
    A selection or target that did not apply holds -1. `actions.engine_action` resolves these
    coordinates to engine Entity ids using the encoder's token ordering.
    """

    x_action: torch.Tensor      # (B, 3) long; action type, state-local selection offset, Monster rank
    x_log_prob: torch.Tensor    # (B, 3) float, 0 where N/A
    x_entropy: torch.Tensor     # (B, 3) float, 0 where N/A
    x_kl_uniform: torch.Tensor  # (B, 3) float, KL-Uniform over legal actions, 0 where N/A


@final
@dataclass
class MapCache:
    """The Map GNN's outputs for every game of a rollout, whose weights don't change.

    The GNN reruns only the games whose Map changed since `x_cell`, and writes their rows in
    place; the others keep theirs. Empty until the rollout's first step. The cache keeps the
    caller's grid without copying it, so a caller must hand a new buffer each step (the Actor's
    snapshots do).
    """

    x_cell: torch.Tensor | None = None     # (E, H * W, NUM_FEATURE_CHANNELS) uint8
    x_hidden: torch.Tensor | None = None   # (E, H * W, dim_map)
    x_readout: torch.Tensor | None = None  # (E, dim_map)


@final
@dataclass
class MapPlan:
    """An update's unique Maps, found once over the states of its rollout and read to the host.

    Its states are the rollout's B states, then its E final states; its U Maps are the distinct
    ones among them, in the order `torch.unique` sorts their packed grids.
    """

    np_state_map: np.ndarray           # (B + E,) int64: each state's Map, by its index among the U
    np_count_room_forward: np.ndarray  # (U, 4) int64: forward Rooms with 0 to 3 children, per Map


@final
@dataclass
class MapDedup:
    """The U Maps a forward runs the Map GNN on, taken on the host from its update's `MapPlan`."""

    x_rep: torch.Tensor                # (U,) long, on the device: each Map's first state
    x_state_map: torch.Tensor          # (B,) long, on the device: each state's Map, by its index
    np_count_room_forward: np.ndarray  # (4,) int64: their forward Rooms with 0 to 3 children


@final
@dataclass
class TokenPlan:
    """An update's unique tokens, found once over the tokens of its rollout and read to the host.

    Its N tokens belong to the rollout's B states followed by its E final states, state after
    state. Its U distinct tokens follow the order in which `torch.unique` sorts their hashes.
    For each unique token, `_gather_tokens` needs its counts of entries, Effects, Effect entries,
    Modifiers and Modifier entries, followed by 1 for a Room token and 0 otherwise.
    """

    np_state_first_token: np.ndarray  # (B + E + 1,) int64: each state's first token, then N
    np_token_unique: np.ndarray       # (N,) int64: each token's unique token, by index among the U
    np_size: np.ndarray               # (U, 6) int64: each unique token's sizes


@final
@dataclass
class TokenDedup:
    """The U tokens a forward's projector runs, taken on the host from its update's `TokenPlan`."""

    x_rep: torch.Tensor           # (U,) long, on the device: each unique token's first token
    x_token_unique: torch.Tensor  # (N,) long, on the device: each token's unique token, by index
    np_size: np.ndarray           # (6,) int64: their sizes, summed


@dataclass
class Rollout:
    """What the Actor collected, before GAE.

    T steps of E games, state t * E + e of the B = T * E states. `states_final` holds each
    game's state after the last step, for the bootstrap value.
    """

    states: States[Live]        # B states
    states_final: States[Live]  # E states

    # Tensors
    x_action: torch.Tensor             # (B, 3): action type, state-local selection offset, Monster rank
    x_log_prob_behavior: torch.Tensor  # (B,) under the weights that played
    x_reward: torch.Tensor             # (T, E, len(RewardStream))
    x_done: torch.Tensor               # (T, E)
    x_num_discount: torch.Tensor       # (T, E) long: k in each step's discount, γ^k


@dataclass
class RunState:
    """What a checkpoint resumes besides the weights, the optimizer and the iteration."""

    total_steps: int          # Steps played so far, every rank's games counted
    ascension_current: float  # The soft-curriculum position
    streak: int               # The consecutive passes at ascension_current so far
