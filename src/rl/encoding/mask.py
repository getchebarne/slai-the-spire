import random
import warnings
from typing import cast

import numpy as np
import slai
import torch
from slai import Card

from src.rl.constants import COLLECTION_KINDS_CARD
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding.focus import derive_focus
from src.rl.types import CollectionKind
from src.rl.types import Token
from src.rl.utils import action_masked


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def encode_batch_masks_into(
    batch_game_state: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    batch_tokens: list[list[Token]],
    np_legal_selection: np.ndarray,
    np_legal_target: np.ndarray,
    np_needs_target: np.ndarray,
    np_legal_action_type: np.ndarray,
) -> None:
    """Reset and write the masks into the supplied token and state rows.

    A legal action names its Entity's one token, found by the Entity's id. An Entity with no
    token, past its collection's cap, is unselectable.

    Parameters
    ----------
    batch_game_state : list[slai.GameState]
        The states, B of them.
    batch_legal_actions : list[list[slai.Action]]
        Each state's legal actions.
    batch_tokens : list[list[Token]]
        Each state's tokens, as (collection kinds, Entity) in window order, one per Entity.
    np_legal_selection : np.ndarray
        (N,) int64. The selectable tokens, a bit per action type, deduped by Card identity among
        the tokens held by the same collections, written in place.
    np_legal_target : np.ndarray
        (N,) int64. The legal Monster targets, a bit per action type, written in place.
    np_needs_target : np.ndarray
        (N,) bool. The tokens whose selection requires a Monster target, written in place.
    np_legal_action_type : np.ndarray
        (B, NUM_ACTION_TYPES) bool. Each state's legal action types, written in place.
    """
    np_legal_selection.fill(0)
    np_legal_target.fill(0)
    np_needs_target.fill(False)
    np_legal_action_type.fill(False)

    row_start = 0
    for idx_state, game_state in enumerate(batch_game_state):
        tokens = batch_tokens[idx_state]

        # Each Entity id's state-local row: its one token's
        row_of_entity_id = {entity.id: row_local for row_local, (_, entity) in enumerate(tokens)}

        # The (`ActionType`, `CollectionKind`s) pairs whose legal ids set a bit, for the dedup
        legal_collection_kinds: set[tuple[int, tuple[CollectionKind, ...]]] = set()

        for action in batch_legal_actions[idx_state]:
            if action_masked(game_state, action):
                continue

            action_type = int(action.action_type)
            ids = action.ids  # Bound once: each read clones the list
            if ids:
                row_local = row_of_entity_id.get(ids[0])
                if row_local is None:
                    # Unselectable Entity (e.g., past its collection's cap)
                    if action_type not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(action_type)
                        warnings.warn(
                            f"Dropped legal action: type={action_type} id {ids[0]} has no token"
                        )
                    continue

                # Entity is selectable for this `ActionType`
                np_legal_selection[row_start + row_local] |= 1 << action_type

                # Add 
                legal_collection_kinds.add((action_type, tokens[row_local][0]))

                # Monster targeting (only relevant for Card play and Potion use)
                if len(ids) == 2:
                    row_local_target = row_of_entity_id[ids[1]]
                    np_needs_target[row_start + row_local] = True
                    np_legal_target[row_start + row_local_target] |= 1 << action_type

            # Mark the `ActionType` as legal for this state
            np_legal_action_type[idx_state, action_type] = True

        # Sanity check: all states should have at least one legal `ActionType`
        if not np_legal_action_type[idx_state].any():
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(game_state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[idx_state]]}"
            )

        # Keep the first legal copy of each Card identity per action type and collection kinds
        for action_type, collection_kinds in legal_collection_kinds:
            if collection_kinds[0] not in COLLECTION_KINDS_CARD:
                continue

            rows = [
                row
                for row, token in enumerate(tokens, row_start)
                if token[0] == collection_kinds and (np_legal_selection[row] >> action_type) & 1
            ]

            # Draw order hidden: scan in an order keyed on the pile, so the kept copy is random
            if CollectionKind.CARD_PILE_DRAW in collection_kinds:
                random.Random(hash(tuple(tokens[row - row_start][1].id for row in rows))).shuffle(
                    rows
                )

            seen_tokens = set()
            for row in rows:
                identity = cast(Card, tokens[row - row_start][1]).identity_hash
                if identity in seen_tokens:
                    # An identical Card is already selectable for this `ActionType`
                    np_legal_selection[row] &= ~(1 << action_type)
                else:
                    seen_tokens.add(identity)

        # Increment `row_start` w/ the number of tokens processed for this state
        row_start += len(tokens)


def unpack_legal_bits(x_legal: torch.Tensor) -> torch.Tensor:
    """Unpack legal masks into one bit per action type.

    Parameters
    ----------
    x_legal : torch.Tensor
        (N,) int64. Each token's legal mask as `encode_batch_masks_into` writes it: bit t set
        where action type t is legal.

    Returns
    -------
    torch.Tensor
        (N, NUM_ACTION_TYPES) bool. Each token's bit of each action type.
    """
    x_idx_action_type = torch.arange(NUM_ACTION_TYPES, device=x_legal.device)

    return ((x_legal.unsqueeze(1) >> x_idx_action_type) & 1).bool()


def _check_action_types() -> None:
    """Check that every action type has its bit in a legal mask's int64 field."""
    if NUM_ACTION_TYPES > 63:
        raise RuntimeError(
            f"The engine has {NUM_ACTION_TYPES} action types: a legal mask holds a bit for each "
            "in a signed int64, 63 at most"
        )


_check_action_types()
