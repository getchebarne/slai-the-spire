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
from src.rl.encoding.card import ENCODING_DIM_CARD_SPARSE
from src.rl.encoding.card import encode_card_into
from src.rl.encoding.card import get_record_key_card
from src.rl.encoding.card import project_outgoing
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import ENCODING_DIM_CHARACTER_SPARSE
from src.rl.encoding.character import encode_character_into
from src.rl.encoding.context import encode_context_into
from src.rl.encoding.effect import ENCODING_DIM_EFFECT_SPARSE
from src.rl.encoding.effect import make_effect_row
from src.rl.encoding.energy import get_energy_max
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION_SPARSE
from src.rl.encoding.event import encode_event_option_into
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK_SPARSE
from src.rl.encoding.incoming import project_incoming
from src.rl.encoding.map_ import encode_map_into
from src.rl.encoding.mask import encode_batch_masks_into
from src.rl.encoding.mask import unpack_legal_bits
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER_SPARSE
from src.rl.encoding.modifier import make_modifier_row
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER_SPARSE
from src.rl.encoding.monster import encode_monster_into
from src.rl.encoding.monster import get_move_effects
from src.rl.encoding.monster import get_move_effects_shown
from src.rl.encoding.monster import get_num_moves
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import ENCODING_DIM_POTION_SPARSE
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.potion import get_record_key_potion
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import ENCODING_DIM_RELIC_SPARSE
from src.rl.encoding.relic import encode_relic_into
from src.rl.encoding.relic import get_record_key_relic
from src.rl.encoding.relic import is_relic_armed
from src.rl.encoding.shared import ENCODING_DIM_OUTGOING_CARD_SPARSE
from src.rl.encoding.shared import ENCODING_DIM_OUTGOING_MONSTER_SPARSE
from src.rl.encoding.shared import ENCODING_DIM_SHARED
from src.rl.encoding.shared import ENCODING_DIM_STASIS_SLOT_SPARSE
from src.rl.encoding.shared import ENCODING_DIM_TAG_SPARSE
from src.rl.encoding.shared import OFF_SHARED_HEALTH_BLOCK
from src.rl.encoding.shared import OFF_SHARED_OUTGOING_HEALTH_LOST
from src.rl.encoding.shared import OFF_SHARED_PRICE
from src.rl.encoding.shared import OFF_SHARED_STASIS_SLOT
from src.rl.encoding.shared import OFF_SHARED_TAG
from src.rl.encoding.shop import ENCODING_DIM_PRICE_SPARSE
from src.rl.encoding.sparse import TABLE_ROW_EMPTY
from src.rl.encoding.sparse import RowTable
from src.rl.encoding.sparse import add_row
from src.rl.encoding.sparse import drop_rows
from src.rl.encoding.sparse import encode_table_rows_into
from src.rl.encoding.sparse import make_row_table
from src.rl.tokens import get_character
from src.rl.tokens import get_entities
from src.rl.tokens import iter_tokens
from src.rl.types import CollectionKind
from src.rl.types import EffectSource
from src.rl.types import EntityKind
from src.rl.types import ModifierOwner
from src.rl.types import OwnedRows
from src.rl.types import RecordKeyCard
from src.rl.types import RecordKeyPotion
from src.rl.types import RecordKeyRelic
from src.rl.types import Reserved
from src.rl.types import Sparse
from src.rl.types import StateCollection
from src.rl.types import States
from src.rl.types import Token
from src.rl.utils import action_masked


# A token's row in the token row table, then its Effect rows and its Modifier rows in theirs
type _Record = tuple[int, Sequence[int], Sequence[int]]


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
COL_START_OUTGOING = _COL_START_SHARED + OFF_SHARED_OUTGOING_HEALTH_LOST

# The range Cards, Relics and Potions share
COL_START_TAG = _COL_START_SHARED + OFF_SHARED_TAG

# Total token encoding dimension
ENCODING_DIM_TOKEN = _COL_START_SHARED + ENCODING_DIM_SHARED

# Most nonzero columns a token row holds
ENCODING_DIM_TOKEN_SPARSE = max(
    ENCODING_DIM_CARD_SPARSE
    + ENCODING_DIM_PRICE_SPARSE
    + ENCODING_DIM_STASIS_SLOT_SPARSE
    + ENCODING_DIM_OUTGOING_CARD_SPARSE
    + ENCODING_DIM_TAG_SPARSE,
    ENCODING_DIM_RELIC_SPARSE + ENCODING_DIM_PRICE_SPARSE + ENCODING_DIM_TAG_SPARSE,
    ENCODING_DIM_POTION_SPARSE + ENCODING_DIM_PRICE_SPARSE + ENCODING_DIM_TAG_SPARSE,
    ENCODING_DIM_MONSTER_SPARSE
    + ENCODING_DIM_HEALTH_BLOCK_SPARSE
    + ENCODING_DIM_STASIS_SLOT_SPARSE
    + ENCODING_DIM_OUTGOING_MONSTER_SPARSE,
    ENCODING_DIM_CHARACTER_SPARSE + ENCODING_DIM_HEALTH_BLOCK_SPARSE,
    ENCODING_DIM_EVENT_OPTION_SPARSE,
)

# The owned row kinds that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[StateCollection] = set()

# Each Entity kind's fill priority: past MAX_EFFECTS or MAX_MODIFIERS, a state keeps the lowest
_FILL_PRIORITY_OF_ENTITY_KIND = {
    EntityKind.CHARACTER: 0,
    EntityKind.EVENT_OPTION: 1,
    EntityKind.MONSTER: 2,
    EntityKind.POTION: 3,
    EntityKind.RELIC: 4,
    EntityKind.CARD: 5,
    EntityKind.ROOM: 6,  # A Room owns no rows
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

# The token rows, each a token's whole row
_ROW_TABLE_TOKEN = make_row_table(ENCODING_DIM_TOKEN_SPARSE)

# The Effect rows, each scaled and with its source's one-hot
_ROW_TABLE_EFFECT = make_row_table(ENCODING_DIM_EFFECT_SPARSE)

# The Modifier rows, each with its owner's one-hot
_ROW_TABLE_MODIFIER = make_row_table(ENCODING_DIM_MODIFIER_SPARSE)

# The kept records of the Cards, Relics and Potions, by their record keys
_RECORD_OF_CARD: dict[RecordKeyCard, _Record] = {}
_RECORD_OF_RELIC: dict[RecordKeyRelic, _Record] = {}
_RECORD_OF_POTION: dict[RecordKeyPotion, _Record] = {}

# Each Effect's row in _ROW_TABLE_EFFECT, under each source: an Effect hashes by field value
_EFFECT_TABLE_ROW_OF_EFFECT: dict[tuple[Effect, EffectSource], int] = {}

# Each Modifier's row in _ROW_TABLE_MODIFIER, under each owner: a Modifier hashes by field value
_MODIFIER_TABLE_ROW_OF_MODIFIER: dict[tuple[Modifier, ModifierOwner], int] = {}

# The holders of the row tables' row indexes, cleared when the row tables are dropped
_TABLE_ROW_HOLDERS: tuple[dict, ...] = (
    _RECORD_OF_CARD,
    _RECORD_OF_RELIC,
    _RECORD_OF_POTION,
    _EFFECT_TABLE_ROW_OF_EFFECT,
    _MODIFIER_TABLE_ROW_OF_MODIFIER,
)

# The row tables are dropped once they hold this many rows between them
_MAX_ROW_TABLE_ROWS = 100_000


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
    np_map_feature = states_reserved.x_map_feature.numpy()
    np_map_topology = states_reserved.x_map_topology.numpy()
    np_context.fill(0)
    for game_state, np_context_row, np_feature, np_topology in zip(
        batch_game_state, np_context, np_map_feature, np_map_topology
    ):
        encode_context_into(game_state, np_context_row)
        encode_map_into(game_state, np_feature, np_topology)

    _encode_batch_effect_pending_into(states_reserved.effect_pending, batch_game_state)

    # The totals last: they total the rows written above
    _encode_totals_into(states_reserved)


def _drop_row_tables_when_full() -> None:
    """Drop the row tables and everything that holds their rows' indexes, once full.

    The token row table fills with the rows made on every write, the Monsters', the Event
    Options' and the Character's, which no later write reads.
    """
    num_rows = (
        _ROW_TABLE_TOKEN.num_rows + _ROW_TABLE_EFFECT.num_rows + _ROW_TABLE_MODIFIER.num_rows
    )
    if num_rows < _MAX_ROW_TABLE_ROWS:
        return

    for table_row_holder in _TABLE_ROW_HOLDERS:
        table_row_holder.clear()

    drop_rows(_ROW_TABLE_TOKEN)
    drop_rows(_ROW_TABLE_EFFECT)
    drop_rows(_ROW_TABLE_MODIFIER)


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

    # Its legal selections: the set bits of its tokens' deduped selection masks
    x_selection_bit = unpack_legal_bits(tokens.x_legal_selection[: len(np_token_state)])
    np_total[:, StateCollection.SELECTIONS] = _sum_counts(
        np_count_token, x_selection_bit.sum(1).numpy()
    )


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

    # Each token's record: its row in the token row table, and the rows it owns in theirs
    batch_record: list[_Record] = []

    # Iterate over states
    for game_state, tokens in zip(batch_game_state, batch_tokens):
        batch_record += _encode_tokens_into(game_state, tokens)

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
    np_entity_kind = x_entity_kind[:num_tokens].numpy()
    np_entity_kind[:] = _NP_ENTITY_KIND_OF_COLLECTION_KIND[np_collection_kind.argmax(1)]

    # Each state's token count, the tokens' rows and the rows the tokens own
    x_token_count[: len(np_count_token)].numpy()[:] = np_count_token
    np_token_table_row = np.array(
        [token_table_row for token_table_row, _, _ in batch_record], dtype=np.intp
    )  # (N_tok,)
    encode_table_rows_into(sparse_token, _ROW_TABLE_TOKEN, np_token_table_row)
    _encode_owned_rows_into(
        effects,
        StateCollection.EFFECTS,
        _ROW_TABLE_EFFECT,
        [effect_table_rows for _, effect_table_rows, _ in batch_record],
        np_count_token,
        np_entity_kind,
        MAX_EFFECTS,
    )
    _encode_owned_rows_into(
        modifiers,
        StateCollection.MODIFIERS,
        _ROW_TABLE_MODIFIER,
        [modifier_table_rows for _, _, modifier_table_rows in batch_record],
        np_count_token,
        np_entity_kind,
        MAX_MODIFIERS,
    )

    return batch_tokens


def _encode_tokens_into(game_state: GameState, tokens: Sequence[Token]) -> list[_Record]:
    """Give each of a state's tokens its record: its row and the rows it owns, in the row tables.

    A Card, Relic or Potion keeps its record under its record key: one lookup finds its row and
    its Effect rows. A Monster's, an Event Option's and the Character's rows read the state
    around them, so their records are made on every write; those rows then stay in the token
    row table, unread, until the row tables are dropped.

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

    Returns
    -------
    list[_Record]
        Each token's row in the token row table, then its Effect rows and its Modifier rows in
        theirs.
    """
    character = get_character(game_state)
    incoming = project_incoming(game_state)
    outgoing_of_card = project_outgoing(game_state)

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
    records: list[_Record] = []

    # Each queued Card's play source: its first CardPlay's
    play_source_of_card_id: dict[int, PlaySource] = {}
    for card_play in game_state.card_play_queue:
        play_source_of_card_id.setdefault(card_play.id_card, card_play.play_source)

    # Iterate over tokens
    for collection_kinds, entity in tokens:
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

                # What playing it would take off each slot's Monster, while it is in the hand
                outgoing = outgoing_of_card.get(entity.id)

                record_key_card = get_record_key_card(
                    entity, energy, draw_below, gold, slot_holder, play_source, outgoing
                )
                record = _RECORD_OF_CARD.get(record_key_card)
                if record is None:
                    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                    encode_card_into(
                        entity,
                        energy,
                        draw_below,
                        gold,
                        slot_holder,
                        play_source,
                        outgoing,
                        np_row[_COL_START_CARD:COL_START_RELIC],
                        np_row[_COL_START_SHARED:],
                    )

                    # Effects
                    effect_table_rows = _get_effect_table_rows(
                        (entity.effects_play, EffectSource.CARD_PLAY),
                        (entity.effects_discard, EffectSource.CARD_DISCARD),
                        (entity.effects_draw, EffectSource.CARD_DRAW),
                    )
                    record = (add_row(_ROW_TABLE_TOKEN, np_row), effect_table_rows, ())
                    _RECORD_OF_CARD[record_key_card] = record

            case Relic():
                linked = entity.id == id_relic_linked
                armed = CollectionKind.RELIC_OWNED in collection_kinds and is_relic_armed(
                    entity, game_state
                )
                record_key_relic = get_record_key_relic(entity, linked, armed, gold)
                record = _RECORD_OF_RELIC.get(record_key_relic)
                if record is None:
                    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                    encode_relic_into(
                        entity,
                        linked,
                        armed,
                        gold,
                        np_row[COL_START_RELIC:COL_START_POTION],
                        np_row[_COL_START_SHARED:],
                    )

                    # Effects
                    effect_table_rows = _get_effect_table_rows(
                        (entity.effects_combat_start, EffectSource.RELIC_COMBAT_START),
                        (entity.effects_turn_start, EffectSource.RELIC_TURN_START),
                        (entity.effects_turn_end, EffectSource.RELIC_TURN_END),
                        (entity.effects_pickup, EffectSource.RELIC_PICKUP),
                        (entity.effects_rest, EffectSource.RELIC_REST),
                        (entity.effects_counter, EffectSource.RELIC_COUNTER),
                        (entity.effects_combat_end, EffectSource.RELIC_COMBAT_END),
                    )
                    record = (add_row(_ROW_TABLE_TOKEN, np_row), effect_table_rows, ())
                    _RECORD_OF_RELIC[record_key_relic] = record

            case Potion():
                # A drink carries its target as a second id; the masks read the Potion's alone
                usable = entity.usable and not action_masked(
                    game_state, Action(ActionType.PotionUse, [entity.id])
                )
                record_key_potion = get_record_key_potion(entity, usable, gold)
                record = _RECORD_OF_POTION.get(record_key_potion)
                if record is None:
                    np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                    encode_potion_into(
                        entity,
                        usable,
                        gold,
                        np_row[COL_START_POTION:COL_START_MONSTER],
                        np_row[_COL_START_SHARED:],
                    )

                    # Effects
                    effect_table_rows = _get_effect_table_rows(
                        (entity.effects, EffectSource.POTION_USE)
                    )
                    record = (add_row(_ROW_TABLE_TOKEN, np_row), effect_table_rows, ())
                    _RECORD_OF_POTION[record_key_potion] = record

            case Monster():
                # What each playable hand Card that hits it would take off it
                outgoing_monster: list[tuple[int, bool]] = []
                for card in get_entities(game_state, outgoing_of_card, Card):
                    outgoing_slot = outgoing_of_card[card.id][entity.slot]
                    if card.playable and outgoing_slot is not None:
                        outgoing_monster.append(outgoing_slot)

                ascension = game_state.ascension
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_monster_into(
                    entity,
                    incoming,
                    outgoing_monster,
                    ascension,
                    np_row[COL_START_MONSTER:COL_START_CHARACTER],
                    np_row[_COL_START_SHARED:],
                )

                # Effects
                effects_sure, effects_possible = get_move_effects_shown(entity, ascension)
                effects_prev_1 = get_move_effects(entity, ascension, entity.move_history[-1:])
                effects_prev_2 = get_move_effects(entity, ascension, entity.move_history[-2:-1])
                idx_moves = range(get_num_moves(entity, ascension))
                effects_moveset = get_move_effects(entity, ascension, idx_moves)
                effect_table_rows = _get_effect_table_rows(
                    (effects_sure, EffectSource.MONSTER_MOVE),
                    (effects_possible, EffectSource.MONSTER_MOVE_POSSIBLE),
                    (effects_prev_1, EffectSource.MONSTER_MOVE_PREV_1),
                    (effects_prev_2, EffectSource.MONSTER_MOVE_PREV_2),
                    (effects_moveset, EffectSource.MONSTER_MOVESET),
                )

                # Modifiers
                modifier_table_rows = [
                    _get_modifier_table_row(modifier, ModifierOwner.MONSTER)
                    for modifier in entity.modifiers
                ]
                record = (
                    add_row(_ROW_TABLE_TOKEN, np_row),
                    effect_table_rows,
                    modifier_table_rows,
                )

            case EventOption():
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_event_option_into(
                    game_state, entity, np_row[COL_START_EVENT_OPTION:_COL_START_SHARED]
                )

                # Everything it does lives in its Effect rows
                effect_table_rows = _get_effect_table_rows(
                    (entity.effects, EffectSource.EVENT_OPTION_SELECT)
                )
                record = (add_row(_ROW_TABLE_TOKEN, np_row), effect_table_rows, ())

            case Character():
                np_row = np.zeros(ENCODING_DIM_TOKEN, dtype=np.float32)
                encode_character_into(
                    game_state,
                    incoming,
                    np_row[COL_START_CHARACTER:COL_START_EVENT_OPTION],
                    np_row[_COL_START_SHARED:],
                )

                # Modifiers
                modifier_table_rows = []
                if game_state.combat is not None:
                    modifier_table_rows = [
                        _get_modifier_table_row(modifier, ModifierOwner.CHARACTER)
                        for modifier in entity.modifiers
                    ]

                record = (add_row(_ROW_TABLE_TOKEN, np_row), (), modifier_table_rows)

            case _:
                # A Room has no columns and owns no rows
                record = (TABLE_ROW_EMPTY, (), ())

        records.append(record)

    return records


def _get_effect_table_rows(*effects_by_source: tuple[Sequence[Effect], EffectSource]) -> list[int]:
    """Find the rows of the Effects a token owns.

    Parameters
    ----------
    *effects_by_source : tuple[Sequence[Effect], EffectSource]
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
    """Find an Effect's row in the Effect row table, adding the row the first time it comes.

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


def _get_modifier_table_row(modifier: Modifier, modifier_owner: ModifierOwner) -> int:
    """Find a Modifier's row in the Modifier row table, adding the row the first time it comes.

    Parameters
    ----------
    modifier : Modifier
        The Modifier.
    modifier_owner : ModifierOwner
        The combatant carrying the Modifier.

    Returns
    -------
    int
        The row's index in _ROW_TABLE_MODIFIER.
    """
    modifier_table_row = _MODIFIER_TABLE_ROW_OF_MODIFIER.get((modifier, modifier_owner))
    if modifier_table_row is not None:
        return modifier_table_row

    modifier_table_row = add_row(_ROW_TABLE_MODIFIER, make_modifier_row(modifier, modifier_owner))
    _MODIFIER_TABLE_ROW_OF_MODIFIER[modifier, modifier_owner] = modifier_table_row

    return modifier_table_row


def _encode_owned_rows_into(
    owned_rows: OwnedRows[Reserved],
    state_collection: StateCollection,
    row_table: RowTable,
    batch_table_rows: list[Sequence[int]],
    np_count_token: np.ndarray,
    np_entity_kind: np.ndarray,
    max_rows: int,
) -> None:
    """Write the rows the tokens own from their row table, at most `max_rows` to a state.

    Parameters
    ----------
    owned_rows : OwnedRows[Reserved]
        One count per token and one row per owned row, written in place.
    state_collection : StateCollection
        The rows' collection, EFFECTS or MODIFIERS, which the overflow warning names.
    row_table : RowTable
        The row table that holds the rows.
    batch_table_rows : list[Sequence[int]]
        Each token's rows in the row table.
    np_count_token : np.ndarray
        (B,) int64. Each state's number of tokens.
    np_entity_kind : np.ndarray
        (N_tok,) int16. Each token's Entity kind, whose fill priority its rows take: past
        `max_rows`, a state keeps the lowest first.
    max_rows : int
        The most rows a state keeps.
    """
    # Each token's count of rows, and each of those rows' index in the row table
    np_count = np.array([len(table_rows) for table_rows in batch_table_rows], dtype=np.int64)
    np_table_row = np.array(list(chain.from_iterable(batch_table_rows)), dtype=np.intp)

    # Cap the rows of the states past `max_rows`
    np_count_row = _sum_counts(np_count_token, np_count)  # (B,)
    if np_count_row.max(initial=0) > max_rows:
        if state_collection not in _WARNED_OVERFLOW:
            _WARNED_OVERFLOW.add(state_collection)
            warnings.warn(
                f"A state's {state_collection.name.capitalize()} exceeded {max_rows}; the last "
                "by fill priority are dropped"
            )

        # Each row's token, and its fill priority: its token's Entity kind's
        np_row_token = np.repeat(np.arange(len(np_count)), np_count)  # (M,)
        np_fill_priority = _NP_FILL_PRIORITY_OF_ENTITY_KIND[np_entity_kind[np_row_token]]

        # The rows first by their fill priority stay, still token after token
        np_keep = np.ones(len(np_table_row), dtype=np.bool_)
        np_row_start = np.cumsum(np_count_row) - np_count_row  # (B,)
        for idx_state in np.flatnonzero(np_count_row > max_rows):
            row_start = np_row_start[idx_state]
            row_stop = row_start + np_count_row[idx_state]
            np_order = np.argsort(np_fill_priority[row_start:row_stop], kind="stable")
            np_keep[row_start + np_order[max_rows:]] = False

        # Each token's rows that stay
        np_count = np.bincount(np_row_token[np_keep], minlength=len(np_count))
        np_table_row = np_table_row[np_keep]

    encode_table_rows_into(owned_rows.sparse, row_table, np_table_row)
    owned_rows.x_count[: len(np_count)].numpy()[:] = np_count


def _check_tables() -> None:
    if set(_FILL_PRIORITY_OF_ENTITY_KIND) != set(EntityKind):
        raise RuntimeError(
            "_FILL_PRIORITY_OF_ENTITY_KIND drifted from EntityKind: "
            f"{set(_FILL_PRIORITY_OF_ENTITY_KIND) ^ set(EntityKind)}"
        )


_check_tables()

# The fill priorities as the table the caps index by Entity kind, once the check has passed
_NP_FILL_PRIORITY_OF_ENTITY_KIND = np.array(
    [_FILL_PRIORITY_OF_ENTITY_KIND[entity_kind] for entity_kind in EntityKind], dtype=np.int64
)
