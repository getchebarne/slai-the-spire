import warnings

import numpy as np
import torch
from slai import GameState

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAX_EFFECTS_STATE
from src.rl.constants import MAX_MODIFIER_ROWS_STATE
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.card import iter_effect_owners as _effect_owners_card
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.effect import OFF_OWNER_CONTEXT
from src.rl.encoding.effect import OwnerContext
from src.rl.encoding.effect import encode_effect_row_into_w_cache
from src.rl.encoding.effect import normalize_effect_rows
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import encode_batch_events
from src.rl.encoding.event import iter_effect_owners as _effect_owners_event
from src.rl.encoding.focus import encode_batch_focus
from src.rl.encoding.layout import EFFECT_SEGMENT_PENDING
from src.rl.encoding.layout import TOKEN_OFFSET
from src.rl.encoding.layout import EntityIdx
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import encode_modifier_row_into
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.monster import iter_effect_owners as _effect_owners_monster
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.potion import iter_effect_owners as _effect_owners_potion
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.relic import iter_effect_owners as _effect_owners_relic
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.shop import encode_batch_shop
from src.rl.types import TEffects
from src.rl.types import TGameState
from src.rl.types import TModifiers
from src.rl.types import TPadded


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL  # thermometer: bit i = level > i

# Fill priority: halt first, then decision contexts, so an overflow only eats deep card piles
_EFFECT_SOURCES = (
    (TOKEN_OFFSET[EntityIdx.EVENT_OPTION], _effect_owners_event, OwnerContext.EVENT_OPTION),
    (TOKEN_OFFSET[EntityIdx.MONSTER], _effect_owners_monster, OwnerContext.MONSTER_MOVE),
    (TOKEN_OFFSET[EntityIdx.POTION], _effect_owners_potion, OwnerContext.POTION),
    (TOKEN_OFFSET[EntityIdx.RELIC], _effect_owners_relic, OwnerContext.RELIC),
    (TOKEN_OFFSET[EntityIdx.CARD], _effect_owners_card, OwnerContext.CARD),
)

# Row families that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()


def encode_batch_game_state(batch_game_state: list[GameState], device: torch.device) -> TGameState:
    # tensorclass batch dim shared by every field
    batch_size = [len(batch_game_state)]

    # Inputs for the bespoke encoders (card/relic/potion encoders pull their own segments)
    batch_monsters = []
    batch_character = []
    batch_energy = []
    batch_map = []
    batch_act = []
    batch_reward = []
    batch_shop = []
    batch_event = []
    batch_deck = []
    batch_health = []
    batch_block = []
    batch_gold = []
    for game_state in batch_game_state:
        combat = game_state.combat
        batch_monsters.append(combat.monsters if combat is not None else [])
        batch_character.append(game_state.character)
        batch_energy.append(combat.energy if combat is not None else None)
        batch_map.append(game_state.map)
        batch_act.append(game_state.act)
        batch_reward.append(game_state.reward)
        batch_shop.append(game_state.shop)
        batch_event.append(game_state.event)
        batch_deck.append(game_state.deck)
        batch_health.append(game_state.character.health)
        batch_block.append(game_state.character.block)
        batch_gold.append(game_state.character.gold)

    # Monsters (emit per-state incoming damage, consumed by the character encoder)
    t_monsters, t_monsters_mask, t_incoming_damages = encode_batch_monsters(
        batch_monsters, batch_health, batch_block, device
    )

    # Reward / shop / event metas + per-item shop prices
    t_reward_meta = encode_batch_rewards(batch_reward, batch_gold, device)
    t_shop_card_prices, t_shop_relic_prices, t_shop_potion_prices, t_shop_meta = encode_batch_shop(
        batch_shop, batch_gold, device
    )
    t_event_meta, t_event_options, t_event_options_mask = encode_batch_events(batch_event, device)

    # Per-type entity tensors (slice layout authored by each encoder)
    t_cards_x, t_cards_mask = encode_batch_cards(batch_game_state, device)
    t_relics_x, t_relics_mask = encode_batch_relics(batch_game_state, device)
    t_potions_x, t_potions_mask = encode_batch_potions(batch_game_state, device)
    t_character_x, t_character_mask = encode_batch_character(
        batch_character, t_incoming_damages, batch_deck, device
    )

    # Map: GNN node-feature grid + next-row room node indices + flat meta
    t_map_grid, t_room_node_idx, t_room_mask, t_map_meta = encode_batch_map(
        batch_map, batch_act, device
    )

    # Deck visibility (hidden in combat) is enforced in encode_batch_cards (np_pad False there)
    return TGameState(
        cards=TPadded(t_cards_x, t_cards_mask, batch_size=batch_size),
        relics=TPadded(t_relics_x, t_relics_mask, batch_size=batch_size),
        potions=TPadded(t_potions_x, t_potions_mask, batch_size=batch_size),
        monsters=TPadded(t_monsters, t_monsters_mask, batch_size=batch_size),
        event_options=TPadded(t_event_options, t_event_options_mask, batch_size=batch_size),
        character=TPadded(t_character_x, t_character_mask, batch_size=batch_size),
        effects=_encode_batch_effects(batch_game_state, device),
        modifiers=_encode_batch_modifiers(batch_game_state, device),
        energy=encode_batch_energy(batch_energy, device),
        map_grid=t_map_grid,
        room_node_idx=t_room_node_idx,
        room_mask=t_room_mask,
        map_meta=t_map_meta,
        focus=encode_batch_focus(batch_game_state, device),
        ascension=_encode_batch_ascension(batch_game_state, device),
        reward_meta=t_reward_meta,
        shop_meta=t_shop_meta,
        event_meta=t_event_meta,
        shop_card_prices=t_shop_card_prices,
        shop_relic_prices=t_shop_relic_prices,
        shop_potion_prices=t_shop_potion_prices,
        batch_size=batch_size,
    )


# Multi-hot vector; bit i = (ascension > i)
def _encode_batch_ascension(
    batch_game_state: list[GameState], device: torch.device
) -> torch.Tensor:
    np_out = np.zeros((len(batch_game_state), ENCODING_DIM_ASCENSION), dtype=np.float32)
    for b, game_state in enumerate(batch_game_state):
        np_out[b, : game_state.ascension] = 1.0
    return torch.from_numpy(np_out).to(device)


def _encode_batch_effects(batch_game_state: list[GameState], device: torch.device) -> TEffects:
    """Pack every state's effect rows into one (B, E, D) block plus owner segments."""
    batch_size = len(batch_game_state)
    np_effect = np.zeros((batch_size, MAX_EFFECTS_STATE, ENCODING_DIM_EFFECT), dtype=np.float32)
    np_segment = np.zeros((batch_size, MAX_EFFECTS_STATE), dtype=np.int64)
    np_mask = np.zeros((batch_size, MAX_EFFECTS_STATE), dtype=bool)

    idx_used = 0
    for b, game_state in enumerate(batch_game_state):
        idx = 0

        # Pending Effect
        pending = game_state.effect_pending
        if pending is not None:
            encode_effect_row_into_w_cache(pending, np_effect[b, idx])
            np_effect[b, idx, OFF_OWNER_CONTEXT + OwnerContext.PENDING] = 1.0
            np_segment[b, idx] = EFFECT_SEGMENT_PENDING
            np_mask[b, idx] = True
            idx += 1

        # Entity Effects
        for offset, iter_effect_owners, context in _EFFECT_SOURCES:
            for slot, effects in iter_effect_owners(game_state):
                for effect in effects:
                    if idx >= MAX_EFFECTS_STATE:
                        if "effects" not in _WARNED_OVERFLOW:
                            _WARNED_OVERFLOW.add("effects")
                            warnings.warn(
                                f"effect rows exceeded {MAX_EFFECTS_STATE} in one state; "
                                "deep card piles dropped (constants.MAX_EFFECTS_STATE)"
                            )
                        break

                    encode_effect_row_into_w_cache(effect, np_effect[b, idx])
                    np_effect[b, idx, OFF_OWNER_CONTEXT + context] = 1.0
                    np_segment[b, idx] = offset + slot
                    np_mask[b, idx] = True
                    idx += 1

        idx_used = max(idx_used, idx)

    # Used prefix only: the tensor keeps its full width (the shard layout freezes
    # shapes at startup), but pads normalize to zero anyway
    normalize_effect_rows(np_effect[:, :idx_used])
    return TEffects(
        x=torch.from_numpy(np_effect).to(device),
        segment=torch.from_numpy(np_segment).to(device),
        mask=torch.from_numpy(np_mask).to(device),
        batch_size=[batch_size],
    )


def _encode_batch_modifiers(batch_game_state: list[GameState], device: torch.device) -> TModifiers:
    """Pack every state's Modifier rows into one (B, N, D) block plus owner segments."""
    batch_size = len(batch_game_state)
    np_mods = np.zeros(
        (batch_size, MAX_MODIFIER_ROWS_STATE, ENCODING_DIM_MODIFIER), dtype=np.float32
    )
    np_segment = np.zeros((batch_size, MAX_MODIFIER_ROWS_STATE), dtype=np.int64)
    np_mask = np.zeros((batch_size, MAX_MODIFIER_ROWS_STATE), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        combat = game_state.combat

        # Collect Modifiers from the Character and Monsters
        owners = [(TOKEN_OFFSET[EntityIdx.CHARACTER], game_state.character.modifiers)]
        if combat is not None:
            owners += [
                (TOKEN_OFFSET[EntityIdx.MONSTER] + i, monster.modifiers)
                for i, monster in enumerate(combat.monsters[:MAX_MONSTERS])
            ]

        idx = 0
        for slot, modifiers in owners:
            for modifier in modifiers:
                if idx >= MAX_MODIFIER_ROWS_STATE:
                    if "modifiers" not in _WARNED_OVERFLOW:
                        _WARNED_OVERFLOW.add("modifiers")
                        warnings.warn(
                            f"modifier rows exceeded {MAX_MODIFIER_ROWS_STATE} in one state; "
                            "dropped (constants.MAX_MODIFIER_ROWS_STATE)"
                        )
                    break

                encode_modifier_row_into(modifier, np_mods[b, idx])
                np_segment[b, idx] = slot
                np_mask[b, idx] = True
                idx += 1

    return TModifiers(
        x=torch.from_numpy(np_mods).to(device),
        segment=torch.from_numpy(np_segment).to(device),
        mask=torch.from_numpy(np_mask).to(device),
        batch_size=[batch_size],
    )
