import warnings
from collections.abc import Sequence
from itertools import chain

import numpy as np
import torch
from slai import Action
from slai import ActionType
from slai import Card
from slai import Character
from slai import Effect
from slai import EventOption
from slai import GameState
from slai import Modifier
from slai import Monster
from slai import PlaySource
from slai import Potion
from slai import Relic

from src.rl.constants import COLLECTION_KINDS_OF_ENTITY_KIND
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import ENCODING_DIM_SPARSE_CARD
from src.rl.encoding.card import encode_card_into
from src.rl.encoding.card import get_record_key_card
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import ENCODING_DIM_SPARSE_CHARACTER
from src.rl.encoding.character import encode_character_into
from src.rl.encoding.context import encode_context_into
from src.rl.encoding.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.effect import make_effect_row
from src.rl.encoding.energy import get_energy_max
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.event import ENCODING_DIM_SPARSE_EVENT_OPTION
from src.rl.encoding.event import encode_event_option_into
from src.rl.encoding.health_block import ENCODING_DIM_SPARSE_HEALTH_BLOCK
from src.rl.encoding.incoming import project_incoming
from src.rl.encoding.map_ import encode_map_grid_into
from src.rl.encoding.mask import encode_batch_masks_into
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.modifier import encode_modifier_row_into
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import ENCODING_DIM_SPARSE_MONSTER
from src.rl.encoding.monster import encode_monster_into
from src.rl.encoding.monster import moveset_effects
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import ENCODING_DIM_SPARSE_POTION
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.potion import get_record_key_potion
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import ENCODING_DIM_SPARSE_RELIC
from src.rl.encoding.relic import encode_relic_into
from src.rl.encoding.relic import get_record_key_relic
from src.rl.encoding.relic import is_relic_armed
from src.rl.encoding.shared import ENCODING_DIM_SHARED
from src.rl.encoding.shared import ENCODING_DIM_SPARSE_STASIS_SLOT
from src.rl.encoding.shared import ENCODING_DIM_SPARSE_TAG
from src.rl.encoding.shared import OFF_SHARED_HEALTH_BLOCK
from src.rl.encoding.shared import OFF_SHARED_PRICE
from src.rl.encoding.shared import OFF_SHARED_STASIS_SLOT
from src.rl.encoding.shared import OFF_SHARED_TAG
from src.rl.encoding.shop import ENCODING_DIM_SPARSE_PRICE
from src.rl.encoding.sparse import TABLE_ROW_EMPTY
from src.rl.encoding.sparse import add_row
from src.rl.encoding.sparse import drop_rows
from src.rl.encoding.sparse import encode_sparse_rows_into
from src.rl.encoding.sparse import encode_table_rows_into
from src.rl.encoding.sparse import make_row_table
from src.rl.encoding.sparse import set_last_rows
from src.rl.tokens import get_character
from src.rl.tokens import iter_tokens
from src.rl.types import CollectionKind
from src.rl.types import EffectSource
from src.rl.types import Entity
from src.rl.types import ModifierOwner
from src.rl.types import OwnedRows
from src.rl.types import Reserved
from src.rl.types import Sparse
from src.rl.types import StateCollection
from src.rl.types import States
from src.rl.types import Token
from src.rl.utils import action_masked


type _ModifierRow = tuple[int, Modifier, ModifierOwner]

# A token's row in the token row table, then its Effect rows in the Effect row table
type _Record = tuple[int, Sequence[int]]


# Each Entity kind's first token column; Rooms have no columns
_COL_START_CARD = 0
COL_START_RELIC = _COL_START_CARD + ENCODING_DIM_CARD
COL_START_POTION = COL_START_RELIC + ENCODING_DIM_RELIC
COL_START_MONSTER = COL_START_POTION + ENCODING_DIM_POTION
COL_START_CHARACTER = COL_START_MONSTER + ENCODING_DIM_MONSTER
COL_START_EVENT_OPTION = COL_START_CHARACTER + ENCODING_DIM_CHARACTER

# The ranges shared across tokens of different Entity kinds
_COL_START_SHARED = COL_START_EVENT_OPTION + ENCODING_DIM_EVENT_OPTION
COL_START_HEALTH_BLOCK = _COL_START_SHARED + OFF_SHARED_HEALTH_BLOCK
COL_START_PRICE = _COL_START_SHARED + OFF_SHARED_PRICE
COL_START_STASIS_SLOT = _COL_START_SHARED + OFF_SHARED_STASIS_SLOT

# The range Cards, Relics and Potions share
COL_START_TAG = _COL_START_SHARED + OFF_SHARED_TAG

# Total token encoding dimension
ENCODING_DIM_TOKEN = _COL_START_SHARED + ENCODING_DIM_SHARED

# Most nonzero columns a token row holds
ENCODING_DIM_SPARSE_TOKEN = max(
    ENCODING_DIM_SPARSE_CARD
    + ENCODING_DIM_SPARSE_PRICE
    + ENCODING_DIM_SPARSE_STASIS_SLOT
    + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_RELIC + ENCODING_DIM_SPARSE_PRICE + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_POTION + ENCODING_DIM_SPARSE_PRICE + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_MONSTER
    + ENCODING_DIM_SPARSE_HEALTH_BLOCK
    + ENCODING_DIM_SPARSE_STASIS_SLOT,
    ENCODING_DIM_SPARSE_CHARACTER + ENCODING_DIM_SPARSE_HEALTH_BLOCK,
    ENCODING_DIM_SPARSE_EVENT_OPTION,
)

# The owned row kinds that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()

# Each Effect source's fill priority: past MAX_EFFECTS, a state keeps the lowest first
_FILL_PRIORITY_OF_EFFECT_SOURCE = {
    EffectSource.EVENT_OPTION_SELECT: 0,
    EffectSource.MONSTER_MOVE: 1,
    EffectSource.POTION_USE: 2,
    EffectSource.RELIC_COMBAT_START: 3,
    EffectSource.RELIC_TURN_START: 3,
    EffectSource.RELIC_TURN_END: 3,
    EffectSource.RELIC_COMBAT_END: 3,
    EffectSource.RELIC_PICKUP: 3,
    EffectSource.RELIC_REST: 3,
    EffectSource.RELIC_COUNTER: 3,
    EffectSource.MONSTER_MOVESET: 4,
    EffectSource.CARD_PLAY: 5,
    EffectSource.CARD_DISCARD: 5,
    EffectSource.CARD_DRAW: 5,
}

# Each collection kind's Entity kind
_ENTITY_KIND_OF_COLLECTION_KIND = {
    collection_kind: entity_kind
    for entity_kind, collection_kinds in COLLECTION_KINDS_OF_ENTITY_KIND.items()
    for collection_kind in collection_kinds
}

# The same, as the table the token writer indexes by collection kind
_NP_ENTITY_KIND_OF_COLLECTION_KIND = np.array(
    [_ENTITY_KIND_OF_COLLECTION_KIND[collection_kind] for collection_kind in CollectionKind],
    dtype=np.int16,
)

# The records' token rows: each a Card's, Relic's or Potion's whole token
_ROW_TABLE_TOKEN = make_row_table(ENCODING_DIM_SPARSE_TOKEN)

# The Effect rows, each scaled and with its source's one-hot
_ROW_TABLE_EFFECT = make_row_table(ENCODING_DIM_SPARSE_EFFECT)

# The kept records of the Cards, Relics and Potions, by their record keys
_RECORD_OF_CARD: dict[
    tuple[int, int, int | None, int | None, int | None, PlaySource | None], _Record
] = {}
_RECORD_OF_RELIC: dict[tuple[int, bool, bool, int | None], _Record] = {}
_RECORD_OF_POTION: dict[tuple[int, bool, int | None], _Record] = {}

# Each Effect's row in _ROW_TABLE_EFFECT, under each source: an Effect hashes by field value
_EFFECT_TABLE_ROW_OF_EFFECT: dict[tuple[Effect, EffectSource], int] = {}

# The holders of the row tables' row indexes: the records and the Effects' table rows
_TABLE_ROW_HOLDERS: tuple[dict, ...] = (
    _RECORD_OF_CARD,
    _RECORD_OF_RELIC,
    _RECORD_OF_POTION,
    _EFFECT_TABLE_ROW_OF_EFFECT,
)

# Both row tables are dropped once the records and the Effects' table rows number this many
_MAX_ROW_TABLE_KEYS = 100_000

# A token's table row while its row is one of the write's dense rows
_TABLE_ROW_DENSE = -1


def encode_states_into(
    states_reserved: States[Reserved],
    batch_game_state: list[GameState],
    batch_legal_actions: list[list[Action]],
) -> None:
    """Encode observations into a buffer's reserved CPU arrays.

    The buffer is invalid while writing. The caller owns exclusive access until the write
    completes and readers have finished with their views. After a failed write the buffer must
    not be read until a later write succeeds. A process runs one write at a time: its writes
    share the row tables.

    Parameters
    ----------
    states_reserved : States[Reserved]
        The reusable CPU capacity, either local or shared with readers.
    batch_game_state : list[GameState]
        The states to encode, exactly the buffer's number of states.
    batch_legal_actions : list[list[Action]]
        Each state's legal actions, in the same order.
    """
    if states_reserved.x_context.ndim != 2:
        raise ValueError("Select one buffer before encoding states")
    
    batch_size = len(states_reserved.x_context)
    if len(batch_game_state) != batch_size or len(batch_legal_actions) != batch_size:
        raise ValueError("States and legal actions must match the buffer's number of states")

    tokens = states_reserved.tokens

    # Full row tables are dropped here, never while a write holds their rows' indexes
    _drop_row_tables_when_full()

    # Tokens
    batch_tokens = _encode_batch_tokens_into(
        x_token_count=tokens.x_count,
        x_collection_kind=tokens.x_collection_kind,
        x_entity_kind=tokens.x_entity_kind,
        sparse_token=tokens.features.sparse,
        effects=tokens.features.effects,
        modifiers=tokens.features.modifiers,
        batch_game_state=batch_game_state,
    )

    # The masks, from each state's legal actions
    num_tokens = sum(map(len, batch_tokens))
    encode_batch_masks_into(
        batch_game_state,
        batch_legal_actions,
        batch_tokens,
        tokens.x_legal_selection[:num_tokens].numpy(),
        tokens.x_legal_target[:num_tokens].numpy(),
        tokens.x_needs_target[:num_tokens].numpy(),
        states_reserved.x_legal_action_type.numpy(),
    )

    # Context and Map
    np_context = states_reserved.x_context.numpy()
    np_map_grid = states_reserved.x_map_grid.numpy()
    np_context.fill(0)
    np_map_grid.fill(0)
    for game_state, np_context_row, np_grid in zip(batch_game_state, np_context, np_map_grid):
        encode_context_into(game_state, np_context_row)
        encode_map_grid_into(game_state, np_grid)

    _encode_batch_effect_pending_into(states_reserved.effect_pending, batch_game_state)

    # The totals last: they total the rows written above
    _encode_totals_into(states_reserved)


def _drop_row_tables_when_full() -> None:
    """Drop both row tables and everything that holds their rows' indexes, once full."""
    num_keys = sum(map(len, _TABLE_ROW_HOLDERS))
    if num_keys < _MAX_ROW_TABLE_KEYS:
        return

    for table_row_holder in _TABLE_ROW_HOLDERS:
        table_row_holder.clear()

    drop_rows(_ROW_TABLE_TOKEN)
    drop_rows(_ROW_TABLE_EFFECT)


def _encode_batch_effect_pending_into(
    effect_pending: Sparse[Reserved],
    batch_game_state: list[GameState],
) -> None:
    """Write each state's pending Effect as one sparse row, empty where no halt is pending.

    Parameters
    ----------
    effect_pending : Sparse[Reserved]
        The reserved per-state rows and their entries.
    batch_game_state : list[GameState]
        The states to encode, B of them.
    """
    batch_effect_table_row = []
    for game_state in batch_game_state:
        effect = game_state.effect_pending
        batch_effect_table_row.append(
            TABLE_ROW_EMPTY
            if effect is None
            else _get_effect_table_row(effect, EffectSource.PENDING)
        )

    encode_table_rows_into(
        effect_pending, _ROW_TABLE_EFFECT, np.array(batch_effect_table_row, dtype=np.intp)
    )


def _encode_totals_into(states_reserved: States[Reserved]) -> None:
    """Write each state's totals from the rows the pass has written.

    Parameters
    ----------
    states_reserved : States[Reserved]
        The reusable CPU capacity, its rows written.
    """
    tokens = states_reserved.tokens
    features = tokens.features
    np_total = states_reserved.x_total.numpy()

    # Its tokens of each collection kind
    np_count_token = tokens.x_count.numpy()  # (B,)
    np_token_state = np.repeat(np.arange(len(np_count_token)), np_count_token)  # (N_tok,)
    np_membership_token, np_membership_collection_kind = (
        tokens.x_collection_kind[: len(np_token_state)].numpy().nonzero()
    )
    np_total[:, : len(CollectionKind)] = np.bincount(
        np_token_state[np_membership_token] * len(CollectionKind) + np_membership_collection_kind,
        minlength=len(np_total) * len(CollectionKind),
    ).reshape(-1, len(CollectionKind))

    # Its tokens, their feature entries, and the rows those tokens own
    np_total[:, StateCollection.TOKENS] = np_count_token
    np_total[:, StateCollection.TOKEN_ENTRIES] = _sum_counts(
        np_count_token, features.sparse.x_count.numpy()
    )
    np_count_effect = _sum_counts(np_count_token, features.effects.x_count.numpy())
    np_count_modifier = _sum_counts(np_count_token, features.modifiers.x_count.numpy())
    np_total[:, StateCollection.EFFECTS] = np_count_effect
    np_total[:, StateCollection.MODIFIERS] = np_count_modifier

    # Each owned row's entries, and the pending Effect's one sparse row per state
    np_total[:, StateCollection.EFFECT_ENTRIES] = _sum_counts(
        np_count_effect, features.effects.sparse.x_count.numpy()
    )
    np_total[:, StateCollection.MODIFIER_ENTRIES] = _sum_counts(
        np_count_modifier, features.modifiers.sparse.x_count.numpy()
    )
    np_total[:, StateCollection.PENDING_ENTRIES] = states_reserved.effect_pending.x_count.numpy()


def _sum_counts(np_count_parent: np.ndarray, np_count_child: np.ndarray) -> np.ndarray:
    """Sum each state's child counts over its consecutive parent rows.

    Parameters
    ----------
    np_count_parent : np.ndarray
        (B,). Each state's number of parent rows.
    np_count_child : np.ndarray
        (parent capacity,). Each parent's number of children; unused capacity is ignored.

    Returns
    -------
    np.ndarray
        (B,) int64. Each state's total number of children.
    """
    np_off = np.concatenate(([0], np.cumsum(np_count_parent, dtype=np.int64)))  # (B + 1,)
    np_total = np.concatenate(([0], np.cumsum(np_count_child[: np_off[-1]], dtype=np.int64)))

    return np.diff(np_total[np_off])


def _encode_batch_tokens_into(
    x_token_count: torch.Tensor,
    x_collection_kind: torch.Tensor,
    x_entity_kind: torch.Tensor,
    sparse_token: Sparse[Reserved],
    effects: OwnedRows[Reserved],
    modifiers: OwnedRows[Reserved],
    batch_game_state: list[GameState],
) -> list[list[Token]]:
    """Write the tokens and their owned rows into the arrays handed in.

    Parameters
    ----------
    x_token_count : torch.Tensor
        (B,) long. Each state's number of tokens, written in place.
    x_collection_kind : torch.Tensor
        (token capacity, len(CollectionKind)) uint8. Each token's collection kinds, 1 at each
        collection holding its Entity, written in place.
    x_entity_kind : torch.Tensor
        (token capacity,) int16. Each token's Entity kind, written in place.
    sparse_token : Sparse[Reserved]
        One row per token.
    effects : OwnedRows[Reserved]
        One count per token and one row per owned Effect.
    modifiers : OwnedRows[Reserved]
        One count per token and one row per owned Modifier.
    batch_game_state : list[GameState]
        The states to encode, B of them.

    Returns
    -------
    list[list[Token]]
        Each state's tokens, as (collection kinds, Entity) in window order, one per Entity.
    """
    # Each state's tokens, as (collection kinds, Entity) in window order
    batch_tokens = [list(iter_tokens(game_state)) for game_state in batch_game_state]

    # Number of tokens for each state in the batch
    np_count_token = np.array([len(tokens) for tokens in batch_tokens], dtype=np.int64)  # (B,)

    # Number of total tokens across all states
    num_tokens = int(np_count_token.sum())
    if num_tokens > len(x_collection_kind):
        raise IndexError("Token count exceeds the destination's capacity")

    # Each token's row in the token row table, and the dense rows of the tokens with no record
    batch_token_table_rows: list[int] = []
    batch_token_rows_dense: list[np.ndarray] = []

    # Initialize lists to store owned rows (Effects, Modifiers)
    batch_token_effect_table_rows: list[Sequence[int]] = []
    batch_modifier_rows: list[_ModifierRow] = []

    # Iterate over states
    for game_state, tokens, row_start in zip(
        batch_game_state, batch_tokens, np.cumsum(np_count_token) - np_count_token
    ):
        (
            token_table_rows,
            token_rows_dense,
            token_effect_table_rows,
            modifier_rows,
        ) = _encode_tokens_into(game_state, tokens, row_start)

        # Accumulate the tokens' table rows and dense rows
        batch_token_table_rows += token_table_rows
        batch_token_rows_dense += token_rows_dense

        # Accumlate Effect and Modifier rows
        batch_token_effect_table_rows += token_effect_table_rows
        batch_modifier_rows += modifier_rows

    # Each token's collection kinds, 1 at each collection holding its Entity
    np_collection_kind = x_collection_kind[:num_tokens].numpy()  # (N_tok, len(CollectionKind))
    np_collection_kind.fill(0)
    np.put(
        np_collection_kind,
        [
            row * len(CollectionKind) + collection_kind
            for row, (collection_kinds, _) in enumerate(chain.from_iterable(batch_tokens))
            for collection_kind in collection_kinds
        ],
        1,
    )

    # Each token's Entity kind, its collection kinds'
    x_entity_kind[:num_tokens].numpy()[:] = _NP_ENTITY_KIND_OF_COLLECTION_KIND[
        np_collection_kind.argmax(1)
    ]

    # Each token's count of Effect rows, and each of those rows' index in the Effect row table
    np_count_effect = np.array(
        [len(effect_table_rows) for effect_table_rows in batch_token_effect_table_rows],
        dtype=np.int64,
    )
    np_effect_table_row = np.array(
        list(chain.from_iterable(batch_token_effect_table_rows)), dtype=np.intp
    )

    # The Modifier rows and each token's count of them
    np_modifier = np.zeros(
        (len(batch_modifier_rows), ENCODING_DIM_MODIFIER), dtype=np.float32
    )
    for np_modifier_row, (_, modifier, owner) in zip(np_modifier, batch_modifier_rows):
        encode_modifier_row_into(modifier, owner, np_modifier_row)
    np_count_modifier = np.bincount(
        [row for row, _, _ in batch_modifier_rows], minlength=num_tokens
    )

    # The dense rows are the token row table's last rows for this write, in token order
    np_token_dense = np.array(batch_token_rows_dense, dtype=np.float32).reshape(
        -1, ENCODING_DIM_TOKEN
    )
    np_token_table_row = np.array(batch_token_table_rows, dtype=np.intp)  # (N_tok,)
    np_token_table_row[np_token_table_row == _TABLE_ROW_DENSE] = set_last_rows(
        _ROW_TABLE_TOKEN, np_token_dense
    )
    encode_table_rows_into(sparse_token, _ROW_TABLE_TOKEN, np_token_table_row)

    # Each state's token count and the rows the tokens own
    x_token_count[: len(np_count_token)].numpy()[:] = np_count_token
    encode_table_rows_into(effects.sparse, _ROW_TABLE_EFFECT, np_effect_table_row)
    effects.x_count[:num_tokens].numpy()[:] = np_count_effect
    encode_sparse_rows_into(modifiers.sparse, np_modifier, ENCODING_DIM_SPARSE_MODIFIER)
    modifiers.x_count[:num_tokens].numpy()[:] = np_count_modifier

    return batch_tokens


def _encode_tokens_into(
    game_state: GameState, tokens: Sequence[Token], row_start: int
) -> tuple[list[int], list[np.ndarray], list[Sequence[int]], list[_ModifierRow]]:
    """Give each of a state's tokens its row and its Effect rows.

    A Card, Relic or Potion reads its record: one lookup for its row in the token row table and
    its Effect rows in the Effect row table. A Monster, the Character and an Event Option get a
    dense row.

    Each token goes at its Entity kind's columns and the shared block that ends the row, both
    written by its kind's encoder; an Event Option writes no shared column. A Room has no
    columns: the model reads the Map network's output there. A token held by several
    collections is written once, with the stasis slot and the play source when the stasis pile or
    the play queue is one of them.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    tokens : Sequence[Token]
        The state's tokens, as (collection kinds, Entity) in window order.
    row_start : int
        The state's first row in the batch.

    Returns
    -------
    token_table_rows : list[int]
        The tokens' rows in the token row table; _TABLE_ROW_DENSE for a token with a dense row.
    token_rows_dense : list[np.ndarray]
        The dense token rows, each (ENCODING_DIM_TOKEN,) float32, token after token.
    token_effect_table_rows : list[Sequence[int]]
        The Effect rows each token owns, at most MAX_EFFECTS over the state: their rows in the
        Effect row table.
    modifier_rows : list[_ModifierRow]
        The Modifier rows the tokens own, token after token, at most MAX_MODIFIERS: each its
        token's batch row, Modifier, and explicit owner kind.
    """
    character = get_character(game_state)
    incoming = project_incoming(game_state)

    # What the Card rows weigh their cost against: outside Combat, a turn's energy
    energy = (
        game_state.combat.energy.energy_current
        if game_state.combat is not None
        else get_energy_max(game_state)
    )
    draw_pile_size = (
        len(game_state.combat.id_card_pile_draw) if game_state.combat is not None else 0
    )
    id_relic_linked = game_state.reward.id_relic_linked if game_state.reward is not None else None
    gold = character.gold
    token_table_rows: list[int] = []
    token_rows_dense: list[np.ndarray] = []
    token_effect_table_rows: list[Sequence[int]] = []
    modifier_rows: list[_ModifierRow] = []

    # Each queued Card's play source: its first CardPlay's
    play_source_of_card_id: dict[int, PlaySource] = {}
    for card_play in game_state.card_play_queue:
        play_source_of_card_id.setdefault(card_play.id_card, card_play.play_source)

    # Iterate over tokens; `row_start` gives the offset this state's tokens start from
    for row_local, (collection_kinds, entity) in enumerate(tokens):
        record = None
        match entity:
            case Card():
                # The Cards below it, when its draw depth shows
                draw_below = (
                    None if entity.draw_depth is None else draw_pile_size - 1 - entity.draw_depth
                )

                # Its holder's slot, while it is in stasis
                slot_holder = None
                if (
                    CollectionKind.CARD_PILE_STASIS in collection_kinds
                    and game_state.combat is not None
                ):
                    slot_holder = game_state.combat.id_card_pile_stasis.index(entity.id)

                # A queued Card plays from its first CardPlay's source
                play_source = None
                if CollectionKind.CARD_PLAY_QUEUE in collection_kinds:
                    play_source = play_source_of_card_id[entity.id]

                record = _get_record_card(
                    entity, energy, draw_below, gold, slot_holder, play_source
                )

            case Relic():
                linked = entity.id == id_relic_linked
                armed = CollectionKind.RELIC_OWNED in collection_kinds and is_relic_armed(
                    entity, game_state
                )
                record = _get_record_relic(entity, linked, armed, gold)

            case Potion():
                # A drink carries its target as a second id; the masks read the Potion's alone
                usable = entity.usable and not action_masked(
                    game_state, Action(ActionType.PotionUse, [entity.id])
                )
                record = _get_record_potion(entity, usable, gold)

            case Monster():
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_monster_into(
                    entity,
                    incoming,
                    game_state.ascension,
                    np_row[COL_START_MONSTER:COL_START_CHARACTER],
                    np_row[_COL_START_SHARED:],
                )

                # Modifiers
                row = row_start + row_local
                modifier_rows += [
                    (row, modifier, ModifierOwner.MONSTER) for modifier in entity.modifiers
                ]

            case EventOption():
                # Everything it does lives in its Effect rows
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_event_option_into(
                    game_state, entity, np_row[COL_START_EVENT_OPTION:_COL_START_SHARED]
                )

            case Character():
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_character_into(
                    game_state,
                    incoming,
                    np_row[COL_START_CHARACTER:COL_START_EVENT_OPTION],
                    np_row[_COL_START_SHARED:],
                )

                # Modifiers
                if game_state.combat is not None:
                    row = row_start + row_local
                    modifier_rows += [
                        (row, modifier, ModifierOwner.CHARACTER) for modifier in entity.modifiers
                    ]

            case _:
                # A Room has no columns and owns no rows
                record = (TABLE_ROW_EMPTY, ())

        # A record gives the token's table row and Effect rows; with none it takes a dense row
        if record is not None:
            table_row, effect_table_rows = record
        else:
            token_rows_dense.append(np_row)
            table_row = _TABLE_ROW_DENSE

            # Effects
            effect_table_rows = _get_effect_table_rows(
                _get_effects_by_source(entity, game_state.ascension)
            )

        token_table_rows.append(table_row)
        token_effect_table_rows.append(effect_table_rows)

    # Cap Effect and Modifier rows and return them
    if sum(map(len, token_effect_table_rows)) > MAX_EFFECTS:
        warning = (
            f"Effect rows exceeded {MAX_EFFECTS} in one state; "
            "deep Card piles dropped (constants.MAX_EFFECTS)"
        )
        if warning not in _WARNED_OVERFLOW:
            _WARNED_OVERFLOW.add(warning)
            warnings.warn(warning)

        # The rows first by their source's fill priority stay, still token after token
        effect_rows = [
            (row_local, effect, effect_source)
            for row_local, (_, entity) in enumerate(tokens)
            for effects, effect_source in _get_effects_by_source(entity, game_state.ascension)
            for effect in effects
        ]
        fill_priorities = [
            _FILL_PRIORITY_OF_EFFECT_SOURCE[effect_source] for _, _, effect_source in effect_rows
        ]
        order = sorted(range(len(effect_rows)), key=fill_priorities.__getitem__)
        effect_rows = [effect_rows[idx_row] for idx_row in sorted(order[:MAX_EFFECTS])]

        # Each token's rows that stay
        effect_table_rows_of_row_local: dict[int, list[int]] = {}
        for row_local, effect, effect_source in effect_rows:
            effect_table_rows_of_row_local.setdefault(row_local, []).append(
                _get_effect_table_row(effect, effect_source)
            )
        token_effect_table_rows = [
            effect_table_rows_of_row_local.get(row_local, ()) for row_local in range(len(tokens))
        ]

    if len(modifier_rows) > MAX_MODIFIERS:
        warning = (
            f"Modifier rows exceeded {MAX_MODIFIERS} in one state; "
            "dropped (constants.MAX_MODIFIERS)"
        )
        if warning not in _WARNED_OVERFLOW:
            _WARNED_OVERFLOW.add(warning)
            warnings.warn(warning)

        # The Character's rows stay first, then the Monsters' by slot
        fill_priorities = [owner != ModifierOwner.CHARACTER for _, _, owner in modifier_rows]
        order = sorted(range(len(modifier_rows)), key=fill_priorities.__getitem__)
        modifier_rows = [modifier_rows[idx_row] for idx_row in sorted(order[:MAX_MODIFIERS])]

    return (
        token_table_rows,
        token_rows_dense,
        token_effect_table_rows,
        modifier_rows,
    )


def _get_record_card(
    card: Card,
    energy: int,
    draw_below: int | None,
    gold: int,
    slot_holder: int | None,
    play_source: PlaySource | None,
) -> _Record:
    """Find a Card's record, making it the first time its key comes.

    Its key is `get_record_key_card`'s and its row is the whole token `encode_card_into` writes.

    Parameters
    ----------
    card : Card
        The Card.
    energy : int
        What its cost weighs against: the Character's energy in Combat, a turn's outside it.
    draw_below : int | None
        The Cards below it in the draw pile; None when its draw depth doesn't show.
    gold : int
        The Character's gold, which its price is read against.
    slot_holder : int | None
        The slot of the Monster holding it in stasis; None out of the stasis pile.
    play_source : PlaySource | None
        Its first CardPlay's source while it is queued; None out of the play queue.

    Returns
    -------
    _Record
        Its row in the token row table, then its Effect rows in the Effect row table.
    """
    record_key_card = get_record_key_card(card, energy, draw_below, gold, slot_holder, play_source)
    record = _RECORD_OF_CARD.get(record_key_card)
    if record is not None:
        return record

    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
    encode_card_into(
        card,
        energy,
        draw_below,
        gold,
        slot_holder,
        play_source,
        np_row[_COL_START_CARD:COL_START_RELIC],
        np_row[_COL_START_SHARED:],
    )

    record = _make_record(np_row, card)
    _RECORD_OF_CARD[record_key_card] = record

    return record


def _get_record_relic(relic: Relic, linked: bool, armed: bool, gold: int) -> _Record:
    """Find a Relic's record, making it the first time its key comes.

    Its key is `get_record_key_relic`'s and its row is the whole token `encode_relic_into` writes.

    Parameters
    ----------
    relic : Relic
        The Relic.
    linked : bool
        Whether the Sapphire Key is linked to it.
    armed : bool
        Whether what an owned Relic waits for holds now (is_relic_armed).
    gold : int
        The Character's gold, which its price is read against.

    Returns
    -------
    _Record
        Its row in the token row table, then its Effect rows in the Effect row table.
    """
    record_key_relic = get_record_key_relic(relic, linked, armed, gold)
    record = _RECORD_OF_RELIC.get(record_key_relic)
    if record is not None:
        return record

    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
    encode_relic_into(
        relic,
        linked,
        armed,
        gold,
        np_row[COL_START_RELIC:COL_START_POTION],
        np_row[_COL_START_SHARED:],
    )

    record = _make_record(np_row, relic)
    _RECORD_OF_RELIC[record_key_relic] = record

    return record


def _get_record_potion(potion: Potion, usable: bool, gold: int) -> _Record:
    """Find a Potion's record, making it the first time its key comes.

    Its key is `get_record_key_potion`'s and its row is the whole token `encode_potion_into`
    writes.

    Parameters
    ----------
    potion : Potion
        The Potion.
    usable : bool
        Whether the policy can drink it: the engine's `Potion.usable` with the masks folded in.
    gold : int
        The Character's gold, which its price is read against.

    Returns
    -------
    _Record
        Its row in the token row table, then its Effect rows in the Effect row table.
    """
    record_key_potion = get_record_key_potion(potion, usable, gold)
    record = _RECORD_OF_POTION.get(record_key_potion)
    if record is not None:
        return record

    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
    encode_potion_into(
        potion,
        usable,
        gold,
        np_row[COL_START_POTION:COL_START_MONSTER],
        np_row[_COL_START_SHARED:],
    )

    record = _make_record(np_row, potion)
    _RECORD_OF_POTION[record_key_potion] = record

    return record


def _make_record(np_row: np.ndarray, entity: Entity) -> _Record:
    """Make a Card's, Relic's or Potion's record, keeping its rows in the row tables.

    Parameters
    ----------
    np_row : np.ndarray
        (ENCODING_DIM_TOKEN,) float32. The Entity's token row.
    entity : Entity
        The Card, Relic or Potion.

    Returns
    -------
    _Record
        The row's index in the token row table, then the Entity's Effect rows' indexes in the
        Effect row table.
    """
    return (
        add_row(_ROW_TABLE_TOKEN, np_row),
        tuple(_get_effect_table_rows(_get_effects_by_source_without_ascension(entity))),
    )


def _get_effect_table_rows(
    effects_by_source: tuple[tuple[Sequence[Effect], EffectSource], ...]
) -> list[int]:
    """Find the rows of the Effects a token owns.

    Parameters
    ----------
    effects_by_source : tuple[tuple[Sequence[Effect], EffectSource], ...]
        The Effects of each source and that source, in row order.

    Returns
    -------
    list[int]
        The rows' indexes in the Effect row table, in row order.
    """
    return [
        _get_effect_table_row(effect, effect_source)
        for effects, effect_source in effects_by_source
        for effect in effects
    ]


def _get_effect_table_row(effect: Effect, effect_source: EffectSource) -> int:
    """Find an Effect's row in the Effect row table, keeping the row the first time it comes.

    Parameters
    ----------
    effect : Effect
        The Effect.
    effect_source : EffectSource
        What applies the Effect.

    Returns
    -------
    int
        The row's index in _ROW_TABLE_EFFECT.
    """
    effect_table_row = _EFFECT_TABLE_ROW_OF_EFFECT.get((effect, effect_source))
    if effect_table_row is not None:
        return effect_table_row

    effect_table_row = add_row(_ROW_TABLE_EFFECT, make_effect_row(effect, effect_source))
    _EFFECT_TABLE_ROW_OF_EFFECT[effect, effect_source] = effect_table_row

    return effect_table_row


def _get_effects_by_source(
    entity: Entity, ascension: int
) -> tuple[tuple[Sequence[Effect], EffectSource], ...]:
    """List the Effects an Entity's token owns, source after source.

    Parameters
    ----------
    entity : Entity
        The token's Entity.
    ascension : int
        The game's ascension, which sets a Monster's moveset.

    Returns
    -------
    tuple[tuple[Sequence[Effect], EffectSource], ...]
        The Effects of each source and that source, in row order; empty for a Character or a Room.
    """
    match entity:
        case Monster():
            return (
                (entity.move_effects, EffectSource.MONSTER_MOVE),
                (moveset_effects(ascension)[entity.name], EffectSource.MONSTER_MOVESET),
            )
        case _:
            return _get_effects_by_source_without_ascension(entity)


def _get_effects_by_source_without_ascension(
    entity: Entity,
) -> tuple[tuple[Sequence[Effect], EffectSource], ...]:
    """List the Effects an Entity's token owns whatever the ascension, source after source.

    Parameters
    ----------
    entity : Entity
        The token's Entity, not a Monster: its moveset changes with the ascension.

    Returns
    -------
    tuple[tuple[Sequence[Effect], EffectSource], ...]
        The Effects of each source and that source, in row order; empty for a Character or a Room.
    """
    match entity:
        case Card():
            return (
                (entity.effects_play, EffectSource.CARD_PLAY),
                (entity.effects_discard, EffectSource.CARD_DISCARD),
                (entity.effects_draw, EffectSource.CARD_DRAW),
            )
        case Relic():
            return (
                (entity.effects_combat_start, EffectSource.RELIC_COMBAT_START),
                (entity.effects_turn_start, EffectSource.RELIC_TURN_START),
                (entity.effects_turn_end, EffectSource.RELIC_TURN_END),
                (entity.effects_pickup, EffectSource.RELIC_PICKUP),
                (entity.effects_rest, EffectSource.RELIC_REST),
                (entity.effects_counter, EffectSource.RELIC_COUNTER),
                (entity.effects_combat_end, EffectSource.RELIC_COMBAT_END),
            )
        case Potion():
            return ((entity.effects, EffectSource.POTION_USE),)
        case EventOption():
            return ((entity.effects, EffectSource.EVENT_OPTION_SELECT),)
        case _:
            return ()


def _check_tables() -> None:
    sources = set(EffectSource) - {EffectSource.PENDING}
    if set(_FILL_PRIORITY_OF_EFFECT_SOURCE) != sources:
        raise RuntimeError(
            "_FILL_PRIORITY_OF_EFFECT_SOURCE drifted from EffectSource: "
            f"{set(_FILL_PRIORITY_OF_EFFECT_SOURCE) ^ sources}"
        )


_check_tables()
