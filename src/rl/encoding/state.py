import numpy as np
import torch
from slai import GameState

from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.effect import encode_batch_effects
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import encode_batch_events
from src.rl.encoding.focus import encode_batch_focus
from src.rl.encoding.health_block import encode_batch_health_block
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.modifier import encode_batch_modifiers
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.shop import encode_batch_prices
from src.rl.encoding.shop import encode_batch_shop_meta
from src.rl.types import EntityKind
from src.rl.types import X_Entity
from src.rl.types import X_GameState


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL


def encode_batch_game_state(game_states: list[GameState], device: torch.device) -> X_GameState:
    batch_size = [len(game_states)]

    # Entity blocks, each packed to a valid prefix, plus the effect rows their tokens
    # own — emitted by the loop that writes the token, so offset == the token's w
    x_cards, np_cards_idxs, effect_rows_cards, price_rows_cards = encode_batch_cards(
        game_states, device
    )
    x_relics, np_relics_idxs, effect_rows_relics, price_rows_relics = encode_batch_relics(
        game_states, device
    )
    x_potions, np_potions_idxs, effect_rows_potions, price_rows_potions = encode_batch_potions(
        game_states, device
    )
    x_monsters, np_monsters_idxs, effect_rows_moves, effect_rows_movesets = encode_batch_monsters(
        game_states, device
    )
    x_event_meta, x_event_options, np_event_idxs, effect_rows_event_options = encode_batch_events(
        game_states, device
    )
    x_character, np_character_idxs = encode_batch_character(game_states, device)

    # Effects, in fill priority: the halt first, then the decision contexts;
    # an overflow eats the static movesets first, then deep card piles
    x_effects = encode_batch_effects(
        game_states,
        (
            (EntityKind.EVENT_OPTION, effect_rows_event_options),
            (EntityKind.MONSTER, effect_rows_moves),
            (EntityKind.POTION, effect_rows_potions),
            (EntityKind.RELIC, effect_rows_relics),
            (EntityKind.MONSTER, effect_rows_movesets),
            (EntityKind.CARD, effect_rows_cards),
        ),
        device,
    )

    # Map: GNN node-feature grid + next-row room node indices + flat meta
    x_map_grid, x_room_node_idx, np_room_idxs, x_map_meta = encode_batch_map(game_states, device)

    # Prices: one row per shop token, emitted by the loop that writes the token
    x_prices = encode_batch_prices(
        game_states,
        (
            (EntityKind.CARD, price_rows_cards),
            (EntityKind.RELIC, price_rows_relics),
            (EntityKind.POTION, price_rows_potions),
        ),
        device,
    )

    return X_GameState(
        cards=_x_entity(x_cards, np_cards_idxs, device),
        relics=_x_entity(x_relics, np_relics_idxs, device),
        potions=_x_entity(x_potions, np_potions_idxs, device),
        monsters=_x_entity(x_monsters, np_monsters_idxs, device),
        event_options=_x_entity(x_event_options, np_event_idxs, device),
        character=_x_entity(x_character, np_character_idxs, device),
        health_block=encode_batch_health_block(game_states, device),
        prices=x_prices,
        effects=x_effects,
        modifiers=encode_batch_modifiers(game_states, device),
        energy=encode_batch_energy(game_states, device),
        map_grid=x_map_grid,
        room_node_idx=x_room_node_idx,
        room_idxs=torch.from_numpy(np_room_idxs).to(device),
        map_meta=x_map_meta,
        focus=encode_batch_focus(game_states, device),
        ascension=_encode_batch_ascension(game_states, device),
        reward_meta=encode_batch_rewards(game_states, device),
        shop_meta=encode_batch_shop_meta(game_states, device),
        event_meta=x_event_meta,
        batch_size=batch_size,
    )


def _x_entity(x_x: torch.Tensor, np_idxs: np.ndarray, device: torch.device) -> X_Entity:
    return X_Entity(x_x, torch.from_numpy(np_idxs).to(device), batch_size=[x_x.shape[0]])


# Multi-Hot vector; bit i = (ascension > i)
def _encode_batch_ascension(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    np_out = np.zeros((len(game_states), ENCODING_DIM_ASCENSION), dtype=np.float32)
    for b, game_state in enumerate(game_states):
        np_out[b, : game_state.ascension] = 1.01
    
    return torch.from_numpy(np_out).to(device)
