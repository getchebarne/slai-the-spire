import random
import warnings
from typing import cast

import numpy as np
import slai
from slai import Card

from src.rl.constants import NUM_ACTION_TYPES
from src.rl.constants import TOKEN_KINDS_CARD
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.focus import derive_focus
from src.rl.tokens import draw_order_visible
from src.rl.types import Token
from src.rl.types import TokenKind
from src.rl.utils import action_masked


def _card_identity(card: Card) -> tuple:
    """Return the Card features and Effects that determine selection equivalence.

    Same-name Cards with different Effects remain distinct, and so do copies at different prices or
    with different cost overrides. The Card object itself is not a suitable key because its hash
    includes its unique arena id.

    Parameters
    ----------
    card : Card
        The Card to key.

    Returns
    -------
    tuple
        The Card feature key, its cost override and price, and its play, discard and draw Effect
        keys.
    """
    return (
        card_key(card),
        card.cost_override,
        card.price,
        effect_key(card.effects_play),
        effect_key(card.effects_discard),
        effect_key(card.effects_draw),
    )


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def encode_batch_masks(
    batch_game_state: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    batch_tokens: list[list[Token]],
    batch_row_of_entity_id: list[dict[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode the masks from each state's legal actions.

    The actions `action_masked` names are engine-legal but masked.

    The token pass gives each state's tokens, as (token kind, Entity) in window order, and each
    Entity id's first state-local row. The state's batch offset is added when writing a mask,
    so the masks use the encoder's placement. An Entity the encoder dropped is unselectable.

    Parameters
    ----------
    batch_game_state : list[slai.GameState]
        The states, B of them.
    batch_legal_actions : list[list[slai.Action]]
        Each state's legal actions.
    batch_tokens : list[list[Token]]
        Each state's tokens, as (token kind, Entity) in window order.
    batch_row_of_entity_id : list[dict[int, int]]
        Each state's Entity ids, each mapped to its first row within that state's tokens.

    Returns
    -------
    np_legal_selection : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The selectable tokens, per action type, deduped by Card
        identity within each token kind. Monster rows are all False.
    np_legal_target : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The legal Monster targets, per action type. All other
        token rows are False.
    np_needs_target : np.ndarray
        (N,) bool. The tokens' needs-target bits: set where a legal action carries a Monster
        target.
    np_legal_action_type : np.ndarray
        (B, NUM_ACTION_TYPES) bool. The states' legal action types.
    """
    num_tokens = sum(len(tokens) for tokens in batch_tokens)

    # Initialize the masks to all-False
    np_legal_selection = np.zeros((num_tokens, NUM_ACTION_TYPES), dtype=np.bool_)
    np_legal_target = np.zeros((num_tokens, NUM_ACTION_TYPES), dtype=np.bool_)
    np_legal_action_type = np.zeros((len(batch_game_state), NUM_ACTION_TYPES), dtype=np.bool_)
    np_needs_target = np.zeros(num_tokens, dtype=np.bool_)

    encode_batch_masks_into(
        batch_game_state,
        batch_legal_actions,
        batch_tokens,
        batch_row_of_entity_id,
        np_legal_selection,
        np_legal_target,
        np_needs_target,
        np_legal_action_type,
    )
    return np_legal_selection, np_legal_target, np_needs_target, np_legal_action_type


def encode_batch_masks_into(
    batch_game_state: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    batch_tokens: list[list[Token]],
    batch_row_of_entity_id: list[dict[int, int]],
    np_legal_selection: np.ndarray,
    np_legal_target: np.ndarray,
    np_needs_target: np.ndarray,
    np_legal_action_type: np.ndarray,
) -> None:
    """Reset and write the masks into the supplied token and state rows.

    Parameters
    ----------
    batch_game_state : list[slai.GameState]
        The states, B of them.
    batch_legal_actions : list[list[slai.Action]]
        Each state's legal actions.
    batch_tokens : list[list[Token]]
        Each state's tokens, as (token kind, Entity) in window order.
    batch_row_of_entity_id : list[dict[int, int]]
        Each state's Entity ids, each mapped to its first state-local token row.
    np_legal_selection : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The selectable tokens, written in place.
    np_legal_target : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The legal Monster targets, written in place.
    np_needs_target : np.ndarray
        (N,) bool. The tokens whose selection requires a Monster target, written in place.
    np_legal_action_type : np.ndarray
        (B, NUM_ACTION_TYPES) bool. Each state's legal action types, written in place.
    """
    np_legal_selection.fill(False)
    np_legal_target.fill(False)
    np_needs_target.fill(False)
    np_legal_action_type.fill(False)

    row_start = 0
    for idx_state, game_state in enumerate(batch_game_state):
        tokens = batch_tokens[idx_state]
        row_of_entity_id = batch_row_of_entity_id[idx_state]

        # The (`ActionType`, `TokenKind`) pairs whose legal ids set a bit, for the dedup
        legal_token_kinds: set[tuple[int, TokenKind]] = set()

        for action in batch_legal_actions[idx_state]:
            if action_masked(game_state, action):
                continue

            action_type = int(action.action_type)
            ids = action.ids  # Bound once: each read clones the list
            if ids:
                row_local = row_of_entity_id.get(ids[0])
                if row_local is None:
                    # Unselectable Entity (e.g., Character). See `src.rl.encoding.state._TOKEN_KINDS_ID_LOOKUP`
                    if action_type not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(action_type)
                        warnings.warn(
                            f"Dropped legal action: type={action_type} id {ids[0]} has no token"
                        )
                    continue

                # Entity is selectable for this `ActionType`
                np_legal_selection[row_start + row_local, action_type] = True

                # Add 
                legal_token_kinds.add((action_type, tokens[row_local][0]))

                # Monster targeting (only relevant for Card play and Potion use)
                if len(ids) == 2:
                    row_local_target = row_of_entity_id[ids[1]]
                    np_needs_target[row_start + row_local] = True
                    np_legal_target[row_start + row_local_target, action_type] = True

            # Mark the `ActionType` as legal for this state
            np_legal_action_type[idx_state, action_type] = True

        # Sanity check: all states should have at least one legal `ActionType`
        if not np_legal_action_type[idx_state].any():
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(game_state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[idx_state]]}"
            )

        # Keep the first legal copy of each Card identity, separately per kind and action type
        for action_type, token_kind in legal_token_kinds:
            if token_kind not in TOKEN_KINDS_CARD:
                continue
            
            # With Frozen Eye, equal Cards at different draw depths are distinct choices
            if token_kind is TokenKind.CARD_PILE_DRAW and draw_order_visible(game_state):
                continue

            rows = [
                row
                for row, (kind, _) in enumerate(tokens, row_start)
                if kind == token_kind and np_legal_selection[row, action_type]
            ]

            # Draw order hidden: scan in an order keyed on the pile, so the kept copy is random
            if token_kind is TokenKind.CARD_PILE_DRAW:
                random.Random(hash(tuple(tokens[row - row_start][1].id for row in rows))).shuffle(
                    rows
                )

            seen_tokens = set()
            for row in rows:
                identity = _card_identity(cast(Card, tokens[row - row_start][1]))
                if identity in seen_tokens:
                    # An identical Card is already selectable for this `ActionType`
                    np_legal_selection[row, action_type] = False
                else:
                    seen_tokens.add(identity)

        # Increment `row_start` w/ the number of tokens processed for this state
        row_start += len(tokens)
