from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from enum import StrEnum
from typing import final

import torch
from slai import ActionType
from slai import Effect

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


# In packing order: the families in window order, each family's token kinds in TOKEN_KINDS_* order
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
    CARD_PILE_STASIS = 11
    CARD_PLAY_QUEUE = 12
    CARD_PILE_NIGHTMARE = 13
    CARD_PILE_DRAW = 14
    CARD_PILE_DISCARD = 15
    CARD_PILE_EXHAUST = 16
    CARD_PILE_DECK = 17
    RELIC_REWARD = 18
    RELIC_SHOP = 19
    RELIC_EVENT_ROLL = 20
    RELIC_OWNED = 21
    POTION_OWNED = 22
    POTION_REWARD = 23
    POTION_SHOP = 24
    POTION_EVENT_ROLL = 25
    MONSTERS = 26
    EVENT_OPTIONS = 27
    CHARACTER = 28
    ROOMS = 29


TOKEN_KINDS_CARD_REWARDS = (  # one token kind per reward bundle (card.py owns their packing)
    TokenKind.CARD_REWARD_0,
    TokenKind.CARD_REWARD_1,
    TokenKind.CARD_REWARD_2,
    TokenKind.CARD_REWARD_3,
    TokenKind.CARD_REWARD_4,
)

# Pack order within the card family: decision pools first, observational piles last
TOKEN_KINDS_CARD = (
    TokenKind.CARD_PILE_HAND,
    TokenKind.CARD_PILE_DISCOVER,
    TokenKind.CARD_PILE_SELECTED,
    *TOKEN_KINDS_CARD_REWARDS,
    TokenKind.CARD_SHOP,
    TokenKind.CARD_PILE_EVENT_ROLL,
    TokenKind.CARD_MATCH_FLIPPED,
    TokenKind.CARD_PILE_STASIS,
    TokenKind.CARD_PLAY_QUEUE,
    TokenKind.CARD_PILE_NIGHTMARE,
    TokenKind.CARD_PILE_DRAW,
    TokenKind.CARD_PILE_DISCARD,
    TokenKind.CARD_PILE_EXHAUST,
    TokenKind.CARD_PILE_DECK,
)

# Pack order within the relic family: decision pools first, the owned relics last
TOKEN_KINDS_RELIC = (
    TokenKind.RELIC_REWARD,
    TokenKind.RELIC_SHOP,
    TokenKind.RELIC_EVENT_ROLL,
    TokenKind.RELIC_OWNED,
)

# Pack order within the potion family; every source is engine-bounded
TOKEN_KINDS_POTION = (
    TokenKind.POTION_OWNED,
    TokenKind.POTION_REWARD,
    TokenKind.POTION_SHOP,
    TokenKind.POTION_EVENT_ROLL,
)

# The single-token-kind families
TOKEN_KINDS_MONSTER = (TokenKind.MONSTERS,)
TOKEN_KINDS_EVENT_OPTION = (TokenKind.EVENT_OPTIONS,)
TOKEN_KINDS_CHARACTER = (TokenKind.CHARACTER,)
TOKEN_KINDS_ROOM = (TokenKind.ROOMS,)

# The families in window order, each its token kinds
_TOKEN_KINDS_OF_FAMILY = (
    TOKEN_KINDS_CARD,
    TOKEN_KINDS_RELIC,
    TOKEN_KINDS_POTION,
    TOKEN_KINDS_MONSTER,
    TOKEN_KINDS_EVENT_OPTION,
    TOKEN_KINDS_CHARACTER,
    TOKEN_KINDS_ROOM,
)
NUM_FAMILIES = len(_TOKEN_KINDS_OF_FAMILY)

# Each token kind's family
FAMILY_OF_TOKEN_KIND: dict[TokenKind, int] = {
    token_kind: family
    for family, token_kinds in enumerate(_TOKEN_KINDS_OF_FAMILY)
    for token_kind in token_kinds
}

# Each token kind's token cap: the slice its encoder applies to that collection
CAP_OF_TOKEN_KIND: dict[TokenKind, int] = {
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
    TokenKind.CARD_MATCH_FLIPPED: 1,  # Match and Keep's face-up card
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


def make_channel_allocator() -> tuple[list[tuple[StrEnum, int]], Callable[..., int]]:
    """A (layout, alloc) pair: alloc(channel, width=1) appends and returns the
    offset, so offsets, widths and the total all derive from one list."""
    layout: list[tuple[StrEnum, int]] = []

    def alloc(channel: StrEnum, width: int = 1) -> int:
        offset = sum(w for _, w in layout)
        layout.append((channel, width))
        return offset

    return layout, alloc


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


# An effect before encoding: the token kind and rank of the token that owns it (which become
# X_OwnedSparse's x_owner_offset), what applies it, and the effect. The rank is the token's index
# among its token kind's tokens in pack order: the same rank `X_Action` and `masks.TokenIndex` use
type EffectRow = tuple[TokenKind, int, EffectSource, Effect]

# One list of EffectRows per state in the batch
type BatchEffectRows = list[list[EffectRow]]

# A shop token's price before encoding: the token kind and rank of the shop token that sells it,
# and its price
type PriceRow = tuple[TokenKind, int, int]

# One list of PriceRows per state in the batch
type BatchPriceRows = list[list[PriceRow]]


@final
@dataclass
class X_Tokens:
    """Every entity token as a row, every state's in window order, state after state; `x_count`
    says how many each state has. A family encoder's holds its family's tokens only, which the
    state encoder lays out by state. K is the rows' most nonzero columns: no fixed width anywhere
    but X_Step's regions. Each row is its nonzero (column, value) entries in column order, then
    (0, 0.0) entries up to K; the entries index the shared column space, each family's columns from
    its COLUMN_START"""

    x_column: torch.Tensor  # (N, K) int16
    x_value: torch.Tensor  # (N, K) float32
    x_token_kind: torch.Tensor  # (N,) int16, which also names the family

    # L2 selection: legal under the action type, deduped by card identity
    x_legal: torch.Tensor  # (N, NUM_ACTION_TYPES) bool

    # selections whose legal actions carry a monster target (cards / potions)
    x_needs_target: torch.Tensor  # (N,) bool

    x_node_idx: torch.Tensor  # (N,) int16: a room token's map node; 0 on every other token
    x_count: torch.Tensor  # (B,) long


@final
@dataclass
class X_OwnedSparse:
    """Rows owned by tokens, one family per X_GameState field, which the projector adds into
    their owners: every state's rows, state after state"""

    x_column: torch.Tensor  # (N, K) int16 — as in X_Tokens
    x_value: torch.Tensor  # (N, K) float32
    x_owner_offset: torch.Tensor  # (N,) long: the owner is the offset-th of its state's tokens
    x_count: torch.Tensor  # (B,) long


@final
@dataclass
class X_EffectPending:
    """The pending halts' effect rows, held sparse, one per state with a halt pending, state
    after state: global context, since no token owns them"""

    x_column: torch.Tensor  # (P, K) int16
    x_value: torch.Tensor  # (P, K) float32
    x_count: torch.Tensor  # (B,) long, 0 or 1


@dataclass
class X_GameState:
    """The game state, as the encoders write it and the model, the rollout and the minibatches
    hold it: each block only its rows, every state's in the block's order, state after state, and
    its sparse entries cut to the most a row uses; each block's `x_count` holds each state's rows
    in it. The per-state fields are (B, ...). src/rl/packed.py chooses its states and joins them"""

    tokens: X_Tokens

    # Rows owned by tokens, each naming its owner by x_owner_offset
    health_block: X_OwnedSparse  # one per Character / Monster
    prices: X_OwnedSparse  # one per Shop token (Card / Relic / Potion)
    effects: X_OwnedSparse  # any number per token
    modifiers: X_OwnedSparse  # any number per Character / Monster

    effect_pending: X_EffectPending
    energy: torch.Tensor  # (B, ENCODING_DIM_ENERGY) float32
    map_grid: torch.Tensor  # (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) float32
    map_meta: torch.Tensor  # (B, ENCODING_DIM_MAP_META) float32
    focus: torch.Tensor  # (B, ENCODING_DIM_FOCUS) float32
    ascension: torch.Tensor  # (B, ENCODING_DIM_ASCENSION) thermometer, bit i = level > i
    reward_meta: torch.Tensor  # (B, ENCODING_DIM_REWARD_META) float32
    shop_meta: torch.Tensor  # (B, ENCODING_DIM_SHOP_META) float32
    event_meta: torch.Tensor  # (B, ENCODING_DIM_EVENT_META) float32
    x_legal_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1


@dataclass
class X_Window:
    """The window packed for the model: only its tokens, every state's in window order, state after
    state, as X_Tokens holds them. Per row: its token kind and family, its masks and its state. Per
    state: its tokens, and per token kind its first row and its tokens; a token kind's tokens sit
    in consecutive rows. Per family: its tokens, from the projector's one read"""

    x_token_kind: torch.Tensor  # (N,) long
    x_family: torch.Tensor  # (N,) long
    x_legal: torch.Tensor  # (N, NUM_ACTION_TYPES) bool
    x_needs_target: torch.Tensor  # (N,) bool
    x_state: torch.Tensor  # (N,) long
    x_count: torch.Tensor  # (B,) long
    x_token_kind_row: torch.Tensor  # (B, len(TokenKind)) long, N where the token kind is absent
    x_token_kind_count: torch.Tensor  # (B, len(TokenKind)) long
    num_tokens_family: list[int]  # NUM_FAMILIES


@dataclass
class X_DrawOrderStates:
    """The states the draw-order layer reads, those whose draw pile shows an order (2 or more draw
    tokens): the ones under Frozen Eye (visible) first. With, from the projector's one read, how
    many are visible, and their rows and pile rows (their draw tokens), visible and in all"""

    x_states: torch.Tensor  # (S,) long
    num_states_visible: int
    num_rows_visible: int
    num_rows: int
    num_pile_visible: int
    num_pile: int


@dataclass
class X_CoreOutput:
    """`global_` and `x_tokens` are the policy heads': the transformer's, but the draw-order
    layer's on the states under Frozen Eye whose draw pile shows an order. `global_value` is the
    value head's: the layer's on every state whose draw pile shows an order, the transformer's
    elsewhere. `window` is the packed window's X_Window."""

    global_: torch.Tensor  # (B, dim_global)
    global_value: torch.Tensor  # (B, dim_global)
    x_tokens: torch.Tensor  # (N, dim_entity) the refined rows
    window: X_Window


@dataclass
class X_Action:
    """The sampled hierarchy per row, engine-facing: `token_kind` and `rank` name the L2
    selection as the rank-th token of its token kind, `target` the L3 selection as the
    rank-th monster token, and masks.engine_action names their entities by id. -1 where a
    level didn't apply."""

    action_type: torch.Tensor  # (B,) long
    token_kind: torch.Tensor  # (B,) long — TokenKind of the L2 selection
    rank: torch.Tensor  # (B,) long
    target: torch.Tensor  # (B,) long
    log_prob: torch.Tensor  # (B, 3) float, 0 where N/A
    entropy: torch.Tensor  # (B, 3) float, 0 where N/A
    kl_uniform: torch.Tensor  # (B, 3) float, KL(uniform‖π) over legal actions, 0 where N/A


@dataclass
class X_Step:
    """One step of every game, shared by the step loop and its workers: the action the step
    loop sampled, then the reward, done flag and state, masks included, the worker wrote back.
    Every field is `num_regions` equal regions, a worker's each: its games' entries, and in every
    block rows for all its games at every cap, which it fills from the region's start"""

    actions: torch.Tensor  # (E, 4) long: action type, token kind, rank, target
    rewards: torch.Tensor  # (E, K)
    dones: torch.Tensor  # (E,)
    game_state: X_GameState  # E games; a region's rows past its games' counts are stale
    num_regions: int


@dataclass
class X_Rollout:
    """What the actor collected, before GAE: T steps of E games, state t * E + e of the
    N = T * E states. The `final_` field is each game's state after the last step, for
    the bootstrap value"""

    game_state: X_GameState  # N states

    # game_state's counts on the CPU, where minibatches sum their sizes
    x_count_cpu: torch.Tensor  # (N, number of blocks) long, X_GameState's blocks in order

    actions: torch.Tensor  # (N, 4): action type, token kind, rank, target
    log_probs_behavior: torch.Tensor  # (N,) under the weights that played
    rewards: torch.Tensor  # (T, E, K)
    dones: torch.Tensor  # (T, E)
    final_game_state: X_GameState  # E states


@dataclass
class X_TrainingBatch:
    """Columnar rollout (N = rollout_length * num_envs states). The full-batch states, masks
    included, are stored once and taken per minibatch; GAE returns/advantages are
    precomputed. Values/returns are per reward stream (K = len(RewardStream)); the policy
    advantage is the per-stream advantages summed, then normalized."""

    game_state: X_GameState  # N states — taken per minibatch (take_states)

    # game_state's counts on the CPU, where minibatches sum their sizes
    x_count_cpu: torch.Tensor  # (N, number of blocks) long, X_GameState's blocks in order

    actions: torch.Tensor  # (N, 4): action type, token kind, rank, target
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
    if len(TOKEN_KINDS_CARD_REWARDS) != MAX_REWARD_CARD_BUNDLES:
        raise RuntimeError(
            f"The engine stages up to {MAX_REWARD_CARD_BUNDLES} reward bundles: "
            "add CARD_REWARD_<b> token kinds"
        )
    if set(CAP_OF_TOKEN_KIND) != set(TokenKind):
        raise RuntimeError(
            "A TokenKind has no entry in CAP_OF_TOKEN_KIND: "
            f"{set(TokenKind) - set(CAP_OF_TOKEN_KIND)}"
        )

    # TokenKind is declared in packing order, valued 0..len-1: the family tables, one after
    # another, list it, every token kind in one table
    token_kinds = [
        token_kind
        for token_kinds_family in _TOKEN_KINDS_OF_FAMILY
        for token_kind in token_kinds_family
    ]
    if token_kinds != list(TokenKind) or token_kinds != list(range(len(token_kinds))):
        names = [token_kind.name for token_kind in token_kinds]
        raise RuntimeError(
            "Declare TokenKind in packing order, valued 0..len-1: the TOKEN_KINDS_* family "
            f"tables, one after another, list {names}"
        )


_check_tables()
