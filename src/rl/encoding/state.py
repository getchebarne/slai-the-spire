import numpy as np
import torch
from slai import GameState

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS_STATE
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MODIFIER_ROWS_STATE
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.encoding.card import ENCODING_SPARSE_CARD
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.character import ENCODING_SPARSE_CHARACTER
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.effect import ENCODING_SPARSE_EFFECT
from src.rl.encoding.effect import encode_batch_effect_pending
from src.rl.encoding.effect import encode_batch_effects
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
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
from src.rl.encoding.monster import ENCODING_SPARSE_MONSTER
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.potion import ENCODING_SPARSE_POTION
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.relic import ENCODING_SPARSE_RELIC
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.shop import ENCODING_SPARSE_PRICE
from src.rl.encoding.shop import encode_batch_prices
from src.rl.encoding.shop import encode_batch_shop_meta
from src.rl.types import CAP_OF_SLICE_KIND
from src.rl.types import SLICE_KINDS_CARD
from src.rl.types import SLICE_KINDS_POTION
from src.rl.types import SLICE_KINDS_RELIC
from src.rl.types import X_EffectPending
from src.rl.types import X_EntitySparse
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL

# The caps of the blocks several collections pack into: the sum of the collections' caps
CAP_CARDS = sum(CAP_OF_SLICE_KIND[slice_kind] for slice_kind in SLICE_KINDS_CARD)
CAP_RELICS = sum(CAP_OF_SLICE_KIND[slice_kind] for slice_kind in SLICE_KINDS_RELIC)
CAP_POTIONS = sum(CAP_OF_SLICE_KIND[slice_kind] for slice_kind in SLICE_KINDS_POTION)
CAP_PRICES = MAX_SIZE_SHOP_CARDS + MAX_SHOP_RELICS + MAX_SHOP_POTIONS


def encode_batch_game_state(game_states: list[GameState], device: torch.device) -> X_GameState:
    batch_size = [len(game_states)]

    # Entity blocks, each packed to a valid prefix, plus the effect and price rows their
    # tokens own — emitted by the loop that writes the token, so each names its slice kind and rank
    x_cards, effect_rows_cards, price_rows_cards = encode_batch_cards(game_states, device)
    x_relics, effect_rows_relics, price_rows_relics = encode_batch_relics(game_states, device)
    x_potions, effect_rows_potions, price_rows_potions = encode_batch_potions(game_states, device)
    x_monsters, effect_rows_moves, effect_rows_movesets = encode_batch_monsters(
        game_states, device
    )
    x_event_meta, x_event_options, effect_rows_event_options = encode_batch_events(
        game_states, device
    )
    x_character = encode_batch_character(game_states, device)

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
        device,
    )

    # Map: GNN node-feature grid + next-row room node indices + flat meta
    x_map_grid, x_room_node_idx, np_room_slice_kind, x_map_meta = encode_batch_map(
        game_states, device
    )

    # Prices: one row per shop token, emitted by the loop that writes the token
    x_prices = encode_batch_prices(
        game_states, (price_rows_cards, price_rows_relics, price_rows_potions), device
    )

    return X_GameState(
        cards=x_cards,
        relics=x_relics,
        potions=x_potions,
        monsters=x_monsters,
        event_options=x_event_options,
        character=x_character,
        health_block=encode_batch_health_block(game_states, device),
        prices=x_prices,
        effects=x_effects,
        effect_pending=encode_batch_effect_pending(game_states, device),
        modifiers=encode_batch_modifiers(game_states, device),
        energy=encode_batch_energy(game_states, device),
        map_grid=x_map_grid,
        room_node_idx=x_room_node_idx,
        room_slice_kind=torch.from_numpy(np_room_slice_kind).to(device),
        map_meta=x_map_meta,
        focus=encode_batch_focus(game_states, device),
        ascension=_encode_batch_ascension(game_states, device),
        reward_meta=encode_batch_rewards(game_states, device),
        shop_meta=encode_batch_shop_meta(game_states, device),
        event_meta=x_event_meta,
        batch_size=batch_size,
    )


# Multi-Hot vector; bit i = (ascension > i)
def _encode_batch_ascension(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    np_out = np.zeros((len(game_states), ENCODING_DIM_ASCENSION), dtype=np.float32)
    for b, game_state in enumerate(game_states):
        np_out[b, : game_state.ascension] = 1.01

    return torch.from_numpy(np_out).to(device)


def empty_game_state(batch_size: int) -> X_GameState:
    """Game states with every block at its cap and nothing in it, padded as the encoders pad"""
    return X_GameState(
        cards=_empty_entity(batch_size, CAP_CARDS, ENCODING_SPARSE_CARD),
        relics=_empty_entity(batch_size, CAP_RELICS, ENCODING_SPARSE_RELIC),
        potions=_empty_entity(batch_size, CAP_POTIONS, ENCODING_SPARSE_POTION),
        monsters=_empty_entity(batch_size, MAX_MONSTERS, ENCODING_SPARSE_MONSTER),
        event_options=_empty_entity(batch_size, MAX_EVENT_OPTIONS, ENCODING_SPARSE_EVENT_OPTION),
        character=_empty_entity(batch_size, 1, ENCODING_SPARSE_CHARACTER),
        health_block=_empty_owned(batch_size, MAX_MONSTERS + 1, ENCODING_SPARSE_HEALTH_BLOCK),
        prices=_empty_owned(batch_size, CAP_PRICES, ENCODING_SPARSE_PRICE),
        effects=_empty_owned(batch_size, MAX_EFFECTS_STATE, ENCODING_SPARSE_EFFECT),
        effect_pending=X_EffectPending(
            x_column=torch.zeros(batch_size, ENCODING_SPARSE_EFFECT, dtype=torch.int16),
            x_value=torch.zeros(batch_size, ENCODING_SPARSE_EFFECT),
            batch_size=[batch_size],
        ),
        modifiers=_empty_owned(batch_size, MAX_MODIFIER_ROWS_STATE, ENCODING_SPARSE_MODIFIER),
        energy=torch.zeros(batch_size, ENCODING_DIM_ENERGY),
        map_grid=torch.zeros(batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        room_node_idx=torch.full((batch_size, MAP_WIDTH), -1, dtype=torch.long),
        room_slice_kind=torch.full((batch_size, MAP_WIDTH), -1, dtype=torch.int16),
        map_meta=torch.zeros(batch_size, ENCODING_DIM_MAP_META),
        focus=torch.zeros(batch_size, ENCODING_DIM_FOCUS),
        ascension=torch.zeros(batch_size, ENCODING_DIM_ASCENSION),
        reward_meta=torch.zeros(batch_size, ENCODING_DIM_REWARD_META),
        shop_meta=torch.zeros(batch_size, ENCODING_DIM_SHOP_META),
        event_meta=torch.zeros(batch_size, ENCODING_DIM_EVENT_META),
        batch_size=[batch_size],
    )


def _empty_entity(batch_size: int, cap: int, num_sparse: int) -> X_EntitySparse:
    return X_EntitySparse(
        torch.zeros(batch_size, cap, num_sparse, dtype=torch.int16),
        torch.zeros(batch_size, cap, num_sparse),
        torch.full((batch_size, cap), -1, dtype=torch.int16),
        batch_size=[batch_size],
    )


def _empty_owned(batch_size: int, cap: int, num_sparse: int) -> X_OwnedSparse:
    return X_OwnedSparse(
        x_column=torch.zeros(batch_size, cap, num_sparse, dtype=torch.int16),
        x_value=torch.zeros(batch_size, cap, num_sparse),
        x_owner_slice_kind=torch.full((batch_size, cap), -1, dtype=torch.long),
        x_owner_rank=torch.zeros(batch_size, cap, dtype=torch.long),
        batch_size=[batch_size],
    )
