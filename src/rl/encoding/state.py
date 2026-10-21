import numpy as np
import torch
from slai import GameState

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS_STATE
from src.rl.constants import MAX_MODIFIER_ROWS_STATE
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import ENCODING_SPARSE_CARD
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import ENCODING_SPARSE_CHARACTER
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.effect import ENCODING_SPARSE_EFFECT
from src.rl.encoding.effect import encode_batch_effect_pending
from src.rl.encoding.effect import encode_batch_effects
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.event import ENCODING_SPARSE_EVENT_OPTION
from src.rl.encoding.event import encode_batch_events
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import encode_batch_focus
from src.rl.encoding.health_block import ENCODING_SPARSE_HEALTH_BLOCK
from src.rl.encoding.health_block import encode_batch_health_block
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.modifier import ENCODING_SPARSE_MODIFIER
from src.rl.encoding.modifier import encode_batch_modifiers
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import ENCODING_SPARSE_MONSTER
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import ENCODING_SPARSE_POTION
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import ENCODING_SPARSE_RELIC
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.shop import ENCODING_SPARSE_PRICE
from src.rl.encoding.shop import encode_batch_prices
from src.rl.encoding.shop import encode_batch_shop_meta
from src.rl.types import CAP_OF_TOKEN_KIND
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import TokenKind
from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL

# Each family's first column in the shared space, the families in window order; the rooms have
# no columns
COLUMN_START_CARD = 0
COLUMN_START_RELIC = COLUMN_START_CARD + ENCODING_DIM_CARD
COLUMN_START_POTION = COLUMN_START_RELIC + ENCODING_DIM_RELIC
COLUMN_START_MONSTER = COLUMN_START_POTION + ENCODING_DIM_POTION
COLUMN_START_EVENT_OPTION = COLUMN_START_MONSTER + ENCODING_DIM_MONSTER
COLUMN_START_CHARACTER = COLUMN_START_EVENT_OPTION + ENCODING_DIM_EVENT_OPTION
ENCODING_DIM_TOKEN = COLUMN_START_CHARACTER + ENCODING_DIM_CHARACTER

# Most nonzero columns a token row holds, whatever its family: the shared step's one width
ENCODING_SPARSE_TOKEN = max(
    ENCODING_SPARSE_CARD,
    ENCODING_SPARSE_RELIC,
    ENCODING_SPARSE_POTION,
    ENCODING_SPARSE_MONSTER,
    ENCODING_SPARSE_EVENT_OPTION,
    ENCODING_SPARSE_CHARACTER,
)

# The caps of the blocks several collections pack into: the sum of the collections' caps
CAP_TOKENS = sum(CAP_OF_TOKEN_KIND.values())
CAP_PRICES = MAX_SIZE_SHOP_CARDS + MAX_SHOP_RELICS + MAX_SHOP_POTIONS


def encode_batch_game_state(game_states: list[GameState], device: torch.device) -> X_GameState:
    # Entity tokens per family, each in its columns of the shared space, with each state's count of
    # them, plus the effect and price rows their tokens own — emitted by the loop that writes the
    # token, so each names its token kind and rank
    x_cards, effect_rows_cards, price_rows_cards = encode_batch_cards(
        game_states, COLUMN_START_CARD, device
    )
    x_relics, effect_rows_relics, price_rows_relics = encode_batch_relics(
        game_states, COLUMN_START_RELIC, device
    )
    x_potions, effect_rows_potions, price_rows_potions = encode_batch_potions(
        game_states, COLUMN_START_POTION, device
    )
    x_monsters, effect_rows_moves, effect_rows_movesets = encode_batch_monsters(
        game_states, COLUMN_START_MONSTER, device
    )
    x_event_meta, x_event_options, effect_rows_event_options = encode_batch_events(
        game_states, COLUMN_START_EVENT_OPTION, device
    )
    x_character = encode_batch_character(game_states, COLUMN_START_CHARACTER, device)

    # Map: GNN node-feature grid + next-row room node indices + flat meta
    x_map_grid, x_room_node_idx, x_count_rooms, x_map_meta = encode_batch_map(game_states, device)

    # The rooms' tokens: no entries, each its node
    num_rooms = len(x_room_node_idx)
    x_rooms = X_Tokens(
        x_column=torch.zeros(num_rooms, 0, dtype=torch.int16, device=device),
        x_value=torch.zeros(num_rooms, 0, device=device),
        x_token_kind=torch.full((num_rooms,), TokenKind.ROOMS, dtype=torch.int16, device=device),
        x_legal=torch.zeros(num_rooms, NUM_ACTION_TYPES, dtype=torch.bool, device=device),
        x_needs_target=torch.zeros(num_rooms, dtype=torch.bool, device=device),
        x_node_idx=x_room_node_idx,
        x_count=x_count_rooms,
    )

    # The tokens: each state's cards, relics, potions, monsters, event options, character and rooms,
    # state after state
    x_tokens = _tokens_by_state(
        [x_cards, x_relics, x_potions, x_monsters, x_event_options, x_character, x_rooms], device
    )

    # Each state's offset of each token kind's first token, which turns an owner's token kind and
    # rank into its offset: a state's tokens are in token kind order, so the state's tokens of the
    # kinds before
    np_state = np.repeat(np.arange(len(game_states)), x_tokens.x_count.numpy())
    np_token_kind_count = np.bincount(
        np_state * len(TokenKind) + x_tokens.x_token_kind.numpy(),
        minlength=len(game_states) * len(TokenKind),
    ).reshape(len(game_states), len(TokenKind))
    np_first_offset = np.cumsum(np_token_kind_count, 1) - np_token_kind_count

    # Effects, in fill priority; an overflow drops the cards first (the deep
    # piles before the hand), then the static movesets
    x_effects = encode_batch_effects(
        game_states,
        (
            effect_rows_event_options,
            effect_rows_moves,
            effect_rows_potions,
            effect_rows_relics,
            effect_rows_movesets,
            effect_rows_cards,
        ),
        np_first_offset,
        device,
    )

    # Prices: one row per shop token, emitted by the loop that writes the token
    x_prices = encode_batch_prices(
        game_states,
        (price_rows_cards, price_rows_relics, price_rows_potions),
        np_first_offset,
        device,
    )

    # The combatants' health/block and modifier rows, and the pending halt's effect row
    x_health_block = encode_batch_health_block(game_states, np_first_offset, device)
    x_effect_pending = encode_batch_effect_pending(game_states, device)
    x_modifiers = encode_batch_modifiers(game_states, np_first_offset, device)

    return X_GameState(
        tokens=x_tokens,
        health_block=x_health_block,
        prices=x_prices,
        effects=x_effects,
        modifiers=x_modifiers,
        effect_pending=x_effect_pending,
        energy=encode_batch_energy(game_states, device),
        map_grid=x_map_grid,
        map_meta=x_map_meta,
        focus=encode_batch_focus(game_states, device),
        ascension=_encode_batch_ascension(game_states, device),
        reward_meta=encode_batch_rewards(game_states, device),
        shop_meta=encode_batch_shop_meta(game_states, device),
        event_meta=x_event_meta,
        x_legal_action_type=torch.zeros(
            len(game_states), NUM_ACTION_TYPES, dtype=torch.bool, device=device
        ),
    )


def _tokens_by_state(families: list[X_Tokens], device: torch.device) -> X_Tokens:
    """The families' tokens laid out by state: each state's tokens family after family, in
    `families`' order, state after state, their entries widened to the widest family's; their
    masks all False, for build_masks to fill"""
    np_count_family = np.stack(
        [x_tokens_of_family.x_count.numpy() for x_tokens_of_family in families]
    )  # (F, B)
    np_count = np_count_family.sum(0)  # (B,)
    num_rows = int(np_count.sum())
    num_entries = max(x_tokens_of_family.x_column.shape[1] for x_tokens_of_family in families)

    # Each family's first row in each state: the state's first row, past its earlier families'
    np_first_row = (np.cumsum(np_count) - np_count) + (
        np.cumsum(np_count_family, 0) - np_count_family
    )  # (F, B)

    np_column = np.zeros((num_rows, num_entries), dtype=np.int16)
    np_value = np.zeros((num_rows, num_entries), dtype=np.float32)
    np_token_kind = np.empty(num_rows, dtype=np.int16)
    np_node_idx = np.empty(num_rows, dtype=np.int16)
    for x_tokens_of_family, np_count_of_family, np_first_row_of_family in zip(
        families, np_count_family, np_first_row
    ):
        # The family's rows, state after state, each state's from its first row there
        np_shift = np_first_row_of_family - (np.cumsum(np_count_of_family) - np_count_of_family)
        np_row = np.arange(len(x_tokens_of_family.x_token_kind)) + np.repeat(
            np_shift, np_count_of_family
        )
        num_entries_of_family = x_tokens_of_family.x_column.shape[1]
        np_column[np_row, :num_entries_of_family] = x_tokens_of_family.x_column.numpy()
        np_value[np_row, :num_entries_of_family] = x_tokens_of_family.x_value.numpy()
        np_token_kind[np_row] = x_tokens_of_family.x_token_kind.numpy()
        np_node_idx[np_row] = x_tokens_of_family.x_node_idx.numpy()

    return X_Tokens(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_token_kind=torch.from_numpy(np_token_kind).to(device),
        x_legal=torch.zeros(num_rows, NUM_ACTION_TYPES, dtype=torch.bool, device=device),
        x_needs_target=torch.zeros(num_rows, dtype=torch.bool, device=device),
        x_node_idx=torch.from_numpy(np_node_idx).to(device),
        x_count=torch.from_numpy(np_count).to(device),
    )


# Multi-Hot vector; bit i = (ascension > i)
def _encode_batch_ascension(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    np_out = np.zeros((len(game_states), ENCODING_DIM_ASCENSION), dtype=np.float32)
    for b, game_state in enumerate(game_states):
        np_out[b, : game_state.ascension] = 1.01

    return torch.from_numpy(np_out).to(device)


def empty_game_state(batch_size: int) -> X_GameState:
    """`batch_size` game states with nothing in them, every block with rows for all of them at
    every cap: the shared step's, which each worker fills from its region's start"""
    num_tokens = batch_size * CAP_TOKENS
    return X_GameState(
        tokens=X_Tokens(
            x_column=torch.zeros(num_tokens, ENCODING_SPARSE_TOKEN, dtype=torch.int16),
            x_value=torch.zeros(num_tokens, ENCODING_SPARSE_TOKEN),
            x_token_kind=torch.zeros(num_tokens, dtype=torch.int16),
            x_legal=torch.zeros(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
            x_needs_target=torch.zeros(num_tokens, dtype=torch.bool),
            x_node_idx=torch.zeros(num_tokens, dtype=torch.int16),
            x_count=torch.zeros(batch_size, dtype=torch.long),
        ),
        health_block=_empty_owned(batch_size, MAX_MONSTERS + 1, ENCODING_SPARSE_HEALTH_BLOCK),
        prices=_empty_owned(batch_size, CAP_PRICES, ENCODING_SPARSE_PRICE),
        effects=_empty_owned(batch_size, MAX_EFFECTS_STATE, ENCODING_SPARSE_EFFECT),
        modifiers=_empty_owned(batch_size, MAX_MODIFIER_ROWS_STATE, ENCODING_SPARSE_MODIFIER),
        effect_pending=X_EffectPending(
            x_column=torch.zeros(batch_size, ENCODING_SPARSE_EFFECT, dtype=torch.int16),
            x_value=torch.zeros(batch_size, ENCODING_SPARSE_EFFECT),
            x_count=torch.zeros(batch_size, dtype=torch.long),
        ),
        energy=torch.zeros(batch_size, ENCODING_DIM_ENERGY),
        map_grid=torch.zeros(batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        map_meta=torch.zeros(batch_size, ENCODING_DIM_MAP_META),
        focus=torch.zeros(batch_size, ENCODING_DIM_FOCUS),
        ascension=torch.zeros(batch_size, ENCODING_DIM_ASCENSION),
        reward_meta=torch.zeros(batch_size, ENCODING_DIM_REWARD_META),
        shop_meta=torch.zeros(batch_size, ENCODING_DIM_SHOP_META),
        event_meta=torch.zeros(batch_size, ENCODING_DIM_EVENT_META),
        x_legal_action_type=torch.zeros(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
    )


def _empty_owned(batch_size: int, cap: int, num_sparse: int) -> X_OwnedSparse:
    num_rows = batch_size * cap
    return X_OwnedSparse(
        x_column=torch.zeros(num_rows, num_sparse, dtype=torch.int16),
        x_value=torch.zeros(num_rows, num_sparse),
        x_owner_offset=torch.zeros(num_rows, dtype=torch.long),
        x_count=torch.zeros(batch_size, dtype=torch.long),
    )
