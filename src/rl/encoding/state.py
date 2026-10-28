import warnings
from collections.abc import Sequence

import numpy as np
import torch
from slai import Card
from slai import Character
from slai import Effect
from slai import EventOption
from slai import GameState
from slai import Modifier
from slai import Monster
from slai import Potion
from slai import Relic
from slai import Room

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import ENCODING_DIM_SPARSE_CARD
from src.rl.encoding.card import encode_card_into_w_cache
from src.rl.encoding.card import get_card_entities
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import ENCODING_DIM_SPARSE_CHARACTER
from src.rl.encoding.character import encode_character_into
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.effect import encode_batch_effect_pending
from src.rl.encoding.effect import encode_effect_row_into_w_cache
from src.rl.encoding.effect import normalize_effect_rows
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.event import encode_batch_event_meta
from src.rl.encoding.event import get_event_option_entities
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import encode_batch_focus
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.health_block import ENCODING_DIM_SPARSE_HEALTH_BLOCK
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.map_ import get_room_entities
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.modifier import encode_modifier_row_into
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import ENCODING_DIM_SPARSE_MONSTER
from src.rl.encoding.monster import encode_monster_into
from src.rl.encoding.monster import get_monster_entities
from src.rl.encoding.monster import moveset_effects
from src.rl.encoding.monster import outgoing_damage
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import ENCODING_DIM_SPARSE_POTION
from src.rl.encoding.potion import encode_potion_into_w_cache
from src.rl.encoding.potion import get_potion_entities
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import ENCODING_DIM_SPARSE_RELIC
from src.rl.encoding.relic import encode_relic_into_w_cache
from src.rl.encoding.relic import get_relic_entities
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.shop import ENCODING_DIM_SPARSE_PRICE
from src.rl.encoding.shop import encode_batch_shop_meta
from src.rl.encoding.shop import encode_price_into
from src.rl.encoding.sparse import compile_sparse_owned
from src.rl.encoding.sparse import compile_sparse_tokens
from src.rl.types import MAP_TOKEN_KIND_CAP
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import TOKEN_KINDS_CARD
from src.rl.types import TOKEN_KINDS_POTION
from src.rl.types import TOKEN_KINDS_RELIC
from src.rl.types import EffectSource
from src.rl.types import TokenKind
from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL

# The stasis slot columns: a one-hot over the monsters' slots in Combat.monsters
ENCODING_DIM_STASIS_SLOT = MAX_MONSTERS
ENCODING_DIM_SPARSE_SLOT = 1

# Each Entity kind's first column in the shared space; Event Options and Rooms have no columns
COL_START_CARD = 0
COL_START_RELIC = COL_START_CARD + ENCODING_DIM_CARD
COL_START_POTION = COL_START_RELIC + ENCODING_DIM_RELIC
COL_START_MONSTER = COL_START_POTION + ENCODING_DIM_POTION
COL_START_CHARACTER = COL_START_MONSTER + ENCODING_DIM_MONSTER

# The ranges shared across tokens of different Entity kinds
COL_START_HEALTH_BLOCK = COL_START_CHARACTER + ENCODING_DIM_CHARACTER
COL_START_PRICE = COL_START_HEALTH_BLOCK + ENCODING_DIM_HEALTH_BLOCK
COL_START_STASIS_SLOT = COL_START_PRICE + ENCODING_DIM_PRICE

# Total token encoding dimension
ENCODING_DIM_TOKEN = COL_START_STASIS_SLOT + ENCODING_DIM_STASIS_SLOT

# Most nonzero columns a token row holds
ENCODING_DIM_SPARSE_TOKEN = max(
    ENCODING_DIM_SPARSE_CARD + ENCODING_DIM_SPARSE_PRICE + ENCODING_DIM_SPARSE_SLOT,
    ENCODING_DIM_SPARSE_RELIC + ENCODING_DIM_SPARSE_PRICE,
    ENCODING_DIM_SPARSE_POTION + ENCODING_DIM_SPARSE_PRICE,
    ENCODING_DIM_SPARSE_MONSTER + ENCODING_DIM_SPARSE_HEALTH_BLOCK + ENCODING_DIM_SPARSE_SLOT,
    ENCODING_DIM_SPARSE_CHARACTER + ENCODING_DIM_SPARSE_HEALTH_BLOCK,
)

# The cap of the block every collection packs into: the sum of the collections' caps
CAP_TOKENS = sum(MAP_TOKEN_KIND_CAP.values())


# An entity behind a token
type Entity = Card | Relic | Potion | Monster | EventOption | Character | Room

# The owned row kinds that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()

# Each effect source's fill priority: past MAX_EFFECTS, a state keeps the lowest first
_MAP_EFFECT_SOURCE_FILL_PRIORITY = {
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


def get_entities_for_token_kind(token_kind: TokenKind, game_state: GameState) -> Sequence[Entity]:
    """The entities of a token kind's tokens in rank order: the encoder's own lists, so an
    entity's id and its token share one placement."""
    if token_kind in TOKEN_KINDS_CARD:
        return get_card_entities(token_kind, game_state)
    if token_kind in TOKEN_KINDS_RELIC:
        return get_relic_entities(token_kind, game_state)
    if token_kind in TOKEN_KINDS_POTION:
        return get_potion_entities(token_kind, game_state)
    match token_kind:
        case TokenKind.MONSTERS:
            return get_monster_entities(game_state)
        case TokenKind.EVENT_OPTIONS:
            return get_event_option_entities(game_state)
        case TokenKind.CHARACTER:
            return [game_state.character]
        case TokenKind.ROOMS:
            return get_room_entities(game_state)
        case _:
            raise ValueError(f"Unsupported `TokenKind`: {token_kind}")


def encode_batch_game_state(game_states: list[GameState], device: torch.device) -> X_GameState:
    # Map: GNN node-feature grid + flat meta
    x_map_grid, x_map_meta = encode_batch_map(game_states, device)

    # The tokens and the rows they own
    x_tokens = _encode_batch_tokens(game_states, device)
    return X_GameState(
        tokens=x_tokens,
        effect_pending=encode_batch_effect_pending(game_states, device),
        energy=encode_batch_energy(game_states, device),
        map_grid=x_map_grid,
        map_meta=x_map_meta,
        focus=encode_batch_focus(game_states, device),
        ascension=_encode_batch_ascension(game_states, device),
        reward_meta=encode_batch_rewards(game_states, device),
        shop_meta=encode_batch_shop_meta(game_states, device),
        event_meta=encode_batch_event_meta(game_states, device),
        x_legal_action_type=torch.zeros(
            len(game_states), NUM_ACTION_TYPES, dtype=torch.bool, device=device
        ),
    )


def _encode_batch_tokens(game_states: list[GameState], device: torch.device) -> X_Tokens:
    """The tokens, each state's in window order, state after state, their masks all False for
    build_masks to fill; and the effect and modifier rows they own"""
    # Each state's tokens, as (token kind, entity) in window order
    batch_tokens = [
        [
            (token_kind, entity)
            for token_kind in TokenKind
            for entity in get_entities_for_token_kind(token_kind, game_state)
        ]
        for game_state in game_states
    ]

    # Number of tokens for each state in the batch
    np_count = np.array([len(tokens) for tokens in batch_tokens], dtype=np.int64)  # (B,)

    # Number of total tokens
    num_tokens = int(np_count.sum())

    # Initialize arrays to store dense token encodings and their kinds
    np_tokens = np.zeros((num_tokens, ENCODING_DIM_TOKEN), dtype=np.float32) # (N_tok, ENCODING_DIM_TOKEN)
    np_token_kinds = np.empty(num_tokens, dtype=np.int16)

    # Initialize an array to store each room token's map node; 0 on every other token
    np_node_idx = np.zeros(num_tokens, dtype=np.int16)

    # Initialize lists to store owned rows (Effects, Modifiers)
    batch_effect_rows: list[tuple[int, Effect, EffectSource]] = []
    batch_modifier_rows: list[tuple[int, Modifier]] = []

    # Iterate over states
    for game_state, tokens, start_idx in zip(
        game_states, batch_tokens, np.cumsum(np_count) - np_count
    ):
        effect_rows, modifier_rows = _encode_tokens_into(
            game_state, tokens, start_idx, np_tokens, np_token_kinds, np_node_idx
        )

        # Accumlate Effect and Modifier rows
        batch_effect_rows += effect_rows
        batch_modifier_rows += modifier_rows

    # The Effect rows scale after they're written: over the whole block at once
    np_effects = np.zeros((len(batch_effect_rows), ENCODING_DIM_EFFECT), dtype=np.float32)
    for np_effect, (_, effect, effect_source) in zip(np_effects, batch_effect_rows):
        encode_effect_row_into_w_cache(effect, effect_source, np_effect)
    normalize_effect_rows(np_effects)
    np_count_effects = np.bincount([row for row, _, _ in batch_effect_rows], minlength=num_tokens)

    # The Modifier rows and each token's count of them
    np_modifiers = np.zeros((len(batch_modifier_rows), ENCODING_DIM_MODIFIER), dtype=np.float32)
    for np_modifier, (_, modifier) in zip(np_modifiers, batch_modifier_rows):
        encode_modifier_row_into(modifier, np_modifier)
    np_count_modifiers = np.bincount([row for row, _ in batch_modifier_rows], minlength=num_tokens)

    # Return sparse representation
    return compile_sparse_tokens(
        np_tokens,
        np_token_kinds,
        np_node_idx,
        np_count,
        ENCODING_DIM_SPARSE_TOKEN,
        device,
        effects=compile_sparse_owned(
            np_effects, np_count_effects.astype(np.int16), ENCODING_DIM_SPARSE_EFFECT, device
        ),
        modifiers=compile_sparse_owned(
            np_modifiers, np_count_modifiers.astype(np.int16), ENCODING_DIM_SPARSE_MODIFIER, device
        ),
    )


def _encode_tokens_into(
    game_state: GameState,
    tokens: Sequence[tuple[TokenKind, Entity]],
    start_idx: int,
    np_tokens: np.ndarray,
    np_token_kinds: np.ndarray,
    np_node_idx: np.ndarray,
) -> tuple[list[tuple[int, Effect, EffectSource]], list[tuple[int, Modifier]]]:
    """Write a state's tokens into their rows from `start_idx` on, each at its entity kind's
    columns, a combatant's health/block at the health/block columns, a shop token's price at the
    price columns, a monster's slot and a stasis card's holder's at the slot columns and a room's
    map node into `np_node_idx`; and return the effect and modifier rows they own, token after
    token, each its token's row, then what the row's writer reads"""
    monsters = get_monster_entities(game_state)
    damage = outgoing_damage(monsters)
    energy_current = (
        game_state.combat.energy.energy_current if game_state.combat is not None else 0
    )
    effect_rows: list[tuple[int, Effect, EffectSource]] = []
    modifier_rows: list[tuple[int, Modifier]] = []

    # The rows at each entity kind's columns, then at the Health/Block, Price and Slot columns
    np_tokens_card = np_tokens[:, COL_START_CARD:COL_START_RELIC]
    np_tokens_relic = np_tokens[:, COL_START_RELIC:COL_START_POTION]
    np_tokens_potion = np_tokens[:, COL_START_POTION:COL_START_MONSTER]
    np_tokens_monster = np_tokens[:, COL_START_MONSTER:COL_START_CHARACTER]
    np_tokens_character = np_tokens[:, COL_START_CHARACTER:COL_START_HEALTH_BLOCK]
    np_tokens_health_block = np_tokens[:, COL_START_HEALTH_BLOCK:COL_START_PRICE]
    np_tokens_price = np_tokens[:, COL_START_PRICE:COL_START_STASIS_SLOT]
    np_tokens_stasis_slot = np_tokens[:, COL_START_STASIS_SLOT:]

    # Iterate over tokens; `start_idx` gives the offset this state starts writing from
    for row, (token_kind, entity) in enumerate(tokens, start_idx):
        np_token_kinds[row] = token_kind
        match entity:
            case Card():
                encode_card_into_w_cache(entity, energy_current, np_tokens_card[row])

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_tokens_price[row]
                    )

                # Effects
                effect_rows += [
                    (row, effect, source)
                    for effects, source in (
                        (entity.effects_play, EffectSource.CARD_PLAY),
                        (entity.effects_discard, EffectSource.CARD_DISCARD),
                        (entity.effects_draw, EffectSource.CARD_DRAW),
                    )
                    for effect in effects
                ]

                # Stasis slot. Shared w/ Monsters
                if token_kind is TokenKind.CARD_PILE_STASIS and game_state.combat is not None:
                    np_tokens_stasis_slot[
                        row, game_state.combat.card_pile_stasis.index(entity)
                    ] = 1.0
            case Relic():
                encode_relic_into_w_cache(entity, np_tokens_relic[row])

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_tokens_price[row]
                    )

                # Effects
                effect_rows += [
                    (row, effect, source)
                    for effects, source in (
                        (entity.effects_combat_start, EffectSource.RELIC_COMBAT_START),
                        (entity.effects_turn_start, EffectSource.RELIC_TURN_START),
                        (entity.effects_turn_end, EffectSource.RELIC_TURN_END),
                        (entity.effects_pickup, EffectSource.RELIC_PICKUP),
                        (entity.effects_rest, EffectSource.RELIC_REST),
                        (entity.effects_counter, EffectSource.RELIC_COUNTER),
                        (entity.effects_combat_end, EffectSource.RELIC_COMBAT_END),
                    )
                    for effect in effects
                ]
            case Potion():
                encode_potion_into_w_cache(entity, np_tokens_potion[row])

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_tokens_price[row]
                    )

                # Effects
                effect_rows += [
                    (row, effect, EffectSource.POTION_USE) for effect in entity.effects
                ]
            case Monster():
                encode_monster_into(
                    entity,
                    game_state.character.health,
                    game_state.character.block,
                    damage,
                    np_tokens_monster[row],
                )

                # Health & Block
                encode_health_block_into(entity.health, entity.block, np_tokens_health_block[row])

                # Stasis slot. Shared w/ Cards in the stasis pile
                np_tokens_stasis_slot[row, monsters.index(entity)] = 1.0

                # Modifiers
                modifier_rows += [(row, modifier) for modifier in entity.modifiers]

                # Effects
                effect_rows += [
                    (row, effect, EffectSource.MONSTER_MOVE) for effect in entity.move_effects
                ]
                effect_rows += [
                    (row, effect, EffectSource.MONSTER_MOVESET)
                    for effect in moveset_effects(game_state.ascension)[entity.name]
                ]
            case EventOption():
                # No columns: everything it does lives in its Effect rows
                effect_rows += [
                    (row, effect, EffectSource.EVENT_OPTION_SELECT) for effect in entity.effects
                ]
            case Character():
                encode_character_into(game_state, damage, np_tokens_character[row])

                # Health & Block
                encode_health_block_into(entity.health, entity.block, np_tokens_health_block[row])

                # Modifiers
                if game_state.combat is not None:
                    modifier_rows += [(row, modifier) for modifier in entity.modifiers]
            case Room():
                # Its map node: no columns, the model reads the map network's output there
                np_node_idx[row] = entity.y * MAP_WIDTH + entity.x

    # Cap Effect and Modifier rows and return them
    if len(effect_rows) > MAX_EFFECTS:
        warning = (
            f"effect rows exceeded {MAX_EFFECTS} in one state; "
            "deep card piles dropped (constants.MAX_EFFECTS)"
        )
        if warning not in _WARNED_OVERFLOW:
            _WARNED_OVERFLOW.add(warning)
            warnings.warn(warning)

        # The rows first by their source's fill priority stay, still token after token
        fill_priorities = [
            _MAP_EFFECT_SOURCE_FILL_PRIORITY[effect_source] for _, _, effect_source in effect_rows
        ]
        order = sorted(range(len(effect_rows)), key=lambda i: fill_priorities[i])
        effect_rows = [effect_rows[i] for i in sorted(order[:MAX_EFFECTS])]

    if len(modifier_rows) > MAX_MODIFIERS:
        warning = (
            f"modifier rows exceeded {MAX_MODIFIERS} in one state; "
            "dropped (constants.MAX_MODIFIERS)"
        )
        if warning not in _WARNED_OVERFLOW:
            _WARNED_OVERFLOW.add(warning)
            warnings.warn(warning)

        # The Character's rows stay first, then the Monsters' by slot
        fill_priorities = [np_token_kinds[row] != TokenKind.CHARACTER for row, _ in modifier_rows]
        order = sorted(range(len(modifier_rows)), key=lambda i: fill_priorities[i])
        modifier_rows = [modifier_rows[i] for i in sorted(order[:MAX_MODIFIERS])]

    return effect_rows, modifier_rows


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
            x_column=torch.zeros(num_tokens, ENCODING_DIM_SPARSE_TOKEN, dtype=torch.int16),
            x_value=torch.zeros(num_tokens, ENCODING_DIM_SPARSE_TOKEN),
            x_token_kind=torch.zeros(num_tokens, dtype=torch.int16),
            x_legal=torch.zeros(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
            x_needs_target=torch.zeros(num_tokens, dtype=torch.bool),
            x_node_idx=torch.zeros(num_tokens, dtype=torch.int16),
            effects=_empty_owned(batch_size, MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT),
            modifiers=_empty_owned(
                batch_size, MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER
            ),
            x_count=torch.zeros(batch_size, dtype=torch.long),
        ),
        effect_pending=X_EffectPending(
            x_column=torch.zeros(batch_size, ENCODING_DIM_SPARSE_EFFECT, dtype=torch.int16),
            x_value=torch.zeros(batch_size, ENCODING_DIM_SPARSE_EFFECT),
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
        x_count=torch.zeros(batch_size * CAP_TOKENS, dtype=torch.int16),
    )


def _check_tables() -> None:
    """Every effect source a token's rows carry has a fill priority"""
    sources = set(EffectSource) - {EffectSource.PENDING}
    if set(_MAP_EFFECT_SOURCE_FILL_PRIORITY) != sources:
        raise RuntimeError(
            "_MAP_EFFECT_SOURCE_FILL_PRIORITY drifted from EffectSource: "
            f"{set(_MAP_EFFECT_SOURCE_FILL_PRIORITY) ^ sources}"
        )


_check_tables()
