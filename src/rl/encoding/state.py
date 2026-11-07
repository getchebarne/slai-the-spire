import warnings
from collections.abc import Sequence

import numpy as np
import torch
from slai import Action
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

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.encoding.card import encode_card_into_w_cache
from src.rl.encoding.character import encode_character_into
from src.rl.encoding.context import encode_context_into
from src.rl.encoding.effect import encode_batch_effect_pending
from src.rl.encoding.effect import encode_effect_row_into_w_cache
from src.rl.encoding.effect import normalize_effect_rows
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.map_ import encode_map_grid_into
from src.rl.encoding.mask import encode_batch_masks
from src.rl.encoding.modifier import encode_modifier_row_into
from src.rl.encoding.monster import encode_monster_into
from src.rl.encoding.monster import get_outgoing_damage
from src.rl.encoding.potion import encode_potion_into_w_cache
from src.rl.encoding.relic import encode_relic_into_w_cache
from src.rl.encoding.shop import encode_price_into
from src.rl.encoding.sparse import encode_sparse
from src.rl.schema.card import TAGS_OF_CARD_NAME
from src.rl.schema.context import ENCODING_DIM_CONTEXT
from src.rl.schema.effect import ENCODING_DIM_EFFECT
from src.rl.schema.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.schema.map_ import NUM_CHANNELS
from src.rl.schema.modifier import ENCODING_DIM_MODIFIER
from src.rl.schema.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.schema.monster import moveset_effects
from src.rl.schema.potion import TAGS_OF_POTION_NAME
from src.rl.schema.relic import TAGS_OF_RELIC_NAME
from src.rl.schema.state import COL_START_CARD
from src.rl.schema.state import COL_START_CHARACTER
from src.rl.schema.state import COL_START_HEALTH_BLOCK
from src.rl.schema.state import COL_START_MONSTER
from src.rl.schema.state import COL_START_POTION
from src.rl.schema.state import COL_START_PRICE
from src.rl.schema.state import COL_START_RELIC
from src.rl.schema.state import COL_START_STASIS_SLOT
from src.rl.schema.state import COL_START_TAG
from src.rl.schema.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.schema.state import ENCODING_DIM_TOKEN
from src.rl.tokens import get_monster_entities
from src.rl.tokens import iter_tokens
from src.rl.types import EffectSource
from src.rl.types import Jagged
from src.rl.types import ModifierOwner
from src.rl.types import StateBatch
from src.rl.types import Token
from src.rl.types import TokenKind
from src.rl.types import Tokens


type EffectRow = tuple[int, Effect, EffectSource]
type ModifierRow = tuple[int, Modifier, ModifierOwner]


# The owned row kinds that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()

# Each Effect source's fill priority: past MAX_EFFECTS, a state keeps the lowest first
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

# The token kinds whose Entity ids the masks look up; no action names the Character. The selected
# pile and the play queue repeat the ids of Cards still in their piles
_TOKEN_KINDS_ID_LOOKUP = frozenset(TokenKind) - {
    TokenKind.CHARACTER,
    TokenKind.CARD_PILE_SELECTED,
    TokenKind.CARD_PLAY_QUEUE,
}


def encode_batch_game_state(
    batch_game_state: list[GameState],
    batch_legal_actions: list[list[Action]],
    device: torch.device,
) -> StateBatch:
    # Tokens
    tokens, x_legal_action_type = _encode_batch_tokens(
        batch_game_state, batch_legal_actions, device
    )

    # Context and Map
    context = np.zeros((len(batch_game_state), ENCODING_DIM_CONTEXT), dtype=np.float32)
    map_grid = np.zeros(
        (len(batch_game_state), MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), dtype=np.float32
    )
    for game_state, context_row, grid in zip(batch_game_state, context, map_grid):
        encode_context_into(game_state, context_row)
        encode_map_grid_into(game_state, grid)

    return StateBatch(
        tokens=tokens,
        pending=encode_batch_effect_pending(batch_game_state, device),
        x_context=torch.from_numpy(context).to(device),
        x_map_grid=torch.from_numpy(map_grid).to(device),
        x_legal_action_type=x_legal_action_type,
    )


def _encode_batch_tokens(
    batch_game_state: list[GameState],
    batch_legal_actions: list[list[Action]],
    device: torch.device,
) -> tuple[Jagged[Tokens], torch.Tensor]:
    """Encode the tokens and each state's legal action types.

    Parameters
    ----------
    batch_game_state : list[GameState]
        The states to encode, B of them.
    batch_legal_actions : list[list[Action]]
        Each state's legal actions.
    device : torch.device
        The device of the returned tensors.

    Returns
    -------
    tokens : Jagged[Tokens]
        The tokens, each state's in window order, state after state, with their masks from each
        state's legal actions, and the Effect and Modifier rows they own.
    x_legal_action_type : torch.Tensor
        (B, NUM_ACTION_TYPES) bool. Each state's legal action types.
    """
    # Each state's tokens, as (token kind, Entity) in window order
    batch_tokens = [list(iter_tokens(game_state)) for game_state in batch_game_state]

    # Number of tokens for each state in the batch
    np_count_tokens = np.array([len(tokens) for tokens in batch_tokens], dtype=np.int64)  # (B,)

    # Number of total tokens across all states
    num_tokens = int(np_count_tokens.sum())

    # Initialize arrays to store dense token encodings and their kinds
    np_token = np.zeros(
        (num_tokens, ENCODING_DIM_TOKEN), dtype=np.float32
    )  # (N_tok, ENCODING_DIM_TOKEN)
    np_token_kind = np.empty(num_tokens, dtype=np.int16)

    # Initialize an array to store each Room token's Map node; 0 on every other token
    np_node_idx = np.zeros(num_tokens, dtype=np.int16)

    # Initialize lists to store owned rows (Effects, Modifiers)
    batch_effect_rows: list[EffectRow] = []
    batch_modifier_rows: list[ModifierRow] = []

    # Each state's Entity ids map to their first state-local token row for the masks
    batch_row_of_entity_id: list[dict[int, int]] = []

    # Iterate over states
    for game_state, tokens, start_idx in zip(
        batch_game_state, batch_tokens, np.cumsum(np_count_tokens) - np_count_tokens
    ):
        effect_rows, modifier_rows, row_of_entity_id = _encode_tokens_into(
            game_state, tokens, start_idx, np_token, np_token_kind, np_node_idx
        )

        # Accumlate Effect and Modifier rows
        batch_effect_rows += effect_rows
        batch_modifier_rows += modifier_rows

        # Accumulate the Entity ids
        batch_row_of_entity_id.append(row_of_entity_id)

    # The Effect rows scale after they're written: over the whole block at once
    np_eff_rows = np.zeros((len(batch_effect_rows), ENCODING_DIM_EFFECT), dtype=np.float32)
    for np_eff_row, (_, effect, effect_source) in zip(np_eff_rows, batch_effect_rows):
        encode_effect_row_into_w_cache(effect, effect_source, np_eff_row)
    normalize_effect_rows(np_eff_rows)
    np_count_eff = np.bincount([row for row, _, _ in batch_effect_rows], minlength=num_tokens)

    # The Modifier rows and each token's count of them
    np_mod_rows = np.zeros(
        (len(batch_modifier_rows), ENCODING_DIM_MODIFIER), dtype=np.float32
    )
    for np_mod_row, (_, modifier, owner) in zip(np_mod_rows, batch_modifier_rows):
        encode_modifier_row_into(modifier, owner, np_mod_row)
    np_count_mod = np.bincount(
        [row for row, _, _ in batch_modifier_rows], minlength=num_tokens
    )

    # The masks, from each state's legal actions
    (
        np_legal_selection,
        np_legal_target,
        np_needs_target,
        np_legal_action_type,
    ) = encode_batch_masks(
        batch_game_state,
        batch_legal_actions,
        batch_tokens,
        batch_row_of_entity_id,
    )

    # Keep ownership boundaries separate from each sparse row's feature-entry lengths.
    tokens = Jagged(
        torch.from_numpy(np_count_tokens).to(device),
        Tokens(
            features=encode_sparse(np_token, ENCODING_DIM_SPARSE_TOKEN, device),
            x_kind=torch.from_numpy(np_token_kind).to(device),
            x_legal_selection=torch.from_numpy(np_legal_selection).to(device),
            x_legal_target=torch.from_numpy(np_legal_target).to(device),
            x_needs_target=torch.from_numpy(np_needs_target).to(device),
            x_node_idx=torch.from_numpy(np_node_idx).to(device),
            effects=Jagged(
                torch.from_numpy(np_count_eff.astype(np.int16)).to(device),
                encode_sparse(np_eff_rows, ENCODING_DIM_SPARSE_EFFECT, device),
            ),
            modifiers=Jagged(
                torch.from_numpy(np_count_mod.astype(np.int16)).to(device),
                encode_sparse(np_mod_rows, ENCODING_DIM_SPARSE_MODIFIER, device),
            ),
        ),
    )
    return tokens, torch.from_numpy(np_legal_action_type).to(device)


def _encode_tokens_into(
    game_state: GameState,
    tokens: Sequence[Token],
    start_idx: int,
    np_token: np.ndarray,
    np_token_kind: np.ndarray,
    np_node_idx: np.ndarray,
) -> tuple[list[EffectRow], list[ModifierRow], dict[int, int]]:
    """Write a state's tokens into their rows from `start_idx` on.

    Each token goes at its Entity kind's columns, a combatant's health/block at the health/block
    columns, a Shop token's price at the price columns, a Monster's slot and a stasis Card's
    holder's at the slot columns, a Card's, Relic's or Potion's Tags at the Tag columns and a
    Room's Map node into `np_node_idx`.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    tokens : Sequence[Token]
        The state's tokens, as (token kind, Entity) in window order.
    start_idx : int
        The state's first row in the batch.
    np_token : np.ndarray
        (N_tok, ENCODING_DIM_TOKEN) float32. The batch's token rows; the state's are written in
        place.
    np_token_kind : np.ndarray
        (N_tok,) int16. The batch's token kinds; the state's are written in place.
    np_node_idx : np.ndarray
        (N_tok,) int16. The batch's tokens' Map nodes; the state's Room tokens' are written in
        place.

    Returns
    -------
    effect_rows : list[EffectRow]
        The Effect rows the tokens own, token after token, at most MAX_EFFECTS: each its token's
        batch row, then what the row's writer reads.
    modifier_rows : list[ModifierRow]
        The Modifier rows the tokens own, token after token, at most MAX_MODIFIERS: each its
        token's batch row, Modifier, and explicit owner kind.
    row_of_entity_id : dict[int, int]
        The first state-local row of each Entity id the masks look up: every token kind's but the
        Character's, the selected pile's and the play queue's (`_TOKEN_KINDS_ID_LOOKUP`).
    """
    monsters = get_monster_entities(game_state)
    damage = get_outgoing_damage(monsters)
    energy_current = (
        game_state.combat.energy.energy_current if game_state.combat is not None else 0
    )
    relic_linked = game_state.reward.relic_linked if game_state.reward is not None else None
    effect_rows: list[EffectRow] = []
    modifier_rows: list[ModifierRow] = []
    row_of_entity_id: dict[int, int] = {}

    # The rows at each Entity kind's columns, then at the Health/Block, Price and Slot columns
    np_token_card = np_token[:, COL_START_CARD:COL_START_RELIC]
    np_token_relic = np_token[:, COL_START_RELIC:COL_START_POTION]
    np_token_potion = np_token[:, COL_START_POTION:COL_START_MONSTER]
    np_token_monster = np_token[:, COL_START_MONSTER:COL_START_CHARACTER]
    np_token_character = np_token[:, COL_START_CHARACTER:COL_START_HEALTH_BLOCK]
    np_token_health_block = np_token[:, COL_START_HEALTH_BLOCK:COL_START_PRICE]
    np_token_price = np_token[:, COL_START_PRICE:COL_START_STASIS_SLOT]
    np_token_stasis_slot = np_token[:, COL_START_STASIS_SLOT:COL_START_TAG]

    # The rows at the Tag columns
    np_token_tag = np_token[:, COL_START_TAG:]

    # Iterate over tokens; `start_idx` gives the offset this state starts writing from
    for row_local, (token_kind, entity) in enumerate(tokens):
        row = start_idx + row_local
        np_token_kind[row] = token_kind

        # Its Entity's id, at the id's first state-local row
        if token_kind in _TOKEN_KINDS_ID_LOOKUP:
            row_of_entity_id.setdefault(entity.id, row_local)

        match entity:
            case Card():
                encode_card_into_w_cache(entity, energy_current, np_token_card[row])

                # Tags. Shared w/ Relics and Potions
                for tag in TAGS_OF_CARD_NAME.get(entity.name, ()):
                    np_token_tag[row, tag] = 1.0

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_token_price[row]
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
                    monster_holder = monsters[game_state.combat.card_pile_stasis.index(entity)]
                    np_token_stasis_slot[row, monster_holder.slot] = 1.0

            case Relic():
                encode_relic_into_w_cache(entity, relic_linked, np_token_relic[row])

                # Tags. Shared w/ Cards and Potions
                for tag in TAGS_OF_RELIC_NAME.get(entity.name, ()):
                    np_token_tag[row, tag] = 1.0

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_token_price[row]
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
                encode_potion_into_w_cache(entity, np_token_potion[row])

                # Tags. Shared w/ Cards and Relics
                for tag in TAGS_OF_POTION_NAME.get(entity.name, ()):
                    np_token_tag[row, tag] = 1.0

                # Shop price
                if entity.price is not None:
                    encode_price_into(
                        entity.price, game_state.character.gold, np_token_price[row]
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
                    game_state.ascension,
                    np_token_monster[row],
                )

                # Health & Block
                encode_health_block_into(entity.health, entity.block, np_token_health_block[row])

                # Stasis slot. Shared w/ Cards in the stasis pile
                np_token_stasis_slot[row, entity.slot] = 1.0

                # Modifiers
                modifier_rows += [
                    (row, modifier, ModifierOwner.MONSTER) for modifier in entity.modifiers
                ]

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
                encode_character_into(game_state, damage, np_token_character[row])

                # Health & Block
                encode_health_block_into(entity.health, entity.block, np_token_health_block[row])

                # Modifiers
                if game_state.combat is not None:
                    modifier_rows += [
                        (row, modifier, ModifierOwner.CHARACTER) for modifier in entity.modifiers
                    ]
            
            case Room():
                # Its Map node: no columns, the model reads the Map network's output there
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
        fill_priorities = [owner != ModifierOwner.CHARACTER for _, _, owner in modifier_rows]
        order = sorted(range(len(modifier_rows)), key=lambda i: fill_priorities[i])
        modifier_rows = [modifier_rows[i] for i in sorted(order[:MAX_MODIFIERS])]

    return effect_rows, modifier_rows, row_of_entity_id


def _check_tables() -> None:
    sources = set(EffectSource) - {EffectSource.PENDING}
    if set(_MAP_EFFECT_SOURCE_FILL_PRIORITY) != sources:
        raise RuntimeError(
            "_MAP_EFFECT_SOURCE_FILL_PRIORITY drifted from EffectSource: "
            f"{set(_MAP_EFFECT_SOURCE_FILL_PRIORITY) ^ sources}"
        )


_check_tables()
