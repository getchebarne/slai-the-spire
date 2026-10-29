import warnings
from typing import cast

import numpy as np
import slai
from slai import ActionType
from slai import Card
from slai import PotionName
from slai import RelicName

from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.focus import draw_order_visible
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import TOKEN_KINDS_CARD
from src.rl.types import Entity
from src.rl.types import TokenKind


def _card_identity_ids(cards: list[Card]) -> list[int]:
    """Return contiguous group ids over a pile; same-identity Cards share an id (selection dedup).

    Groups by `card_key` and the `effect_key` of the Card's play, discard and draw Effects — the
    Card row plus the Effect rows, the Card's full model-visible identity. Two same-name Cards can
    carry different Effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    selections. Not keyed on the Card object: the engine's __hash__ covers the arena `id`, which
    would make every Card distinct and the dedup a no-op.

    Parameters
    ----------
    cards : list[Card]
        The pile's Cards.

    Returns
    -------
    list[int]
        One group id per Card, in pile order, numbered from 0 in first-seen order.
    """
    seen: dict[tuple, int] = {}
    return [
        seen.setdefault(
            (
                card_key(card),
                effect_key(card.effects_play),
                effect_key(card.effects_discard),
                effect_key(card.effects_draw),
            ),
            len(seen),
        )
        for card in cards
    ]


def _pile_ids(
    game_state: slai.GameState, token_kind: TokenKind, entities: list[Entity]
) -> list[int]:
    """Return the identity group-ids for a token kind's dedup.

    Parameters
    ----------
    game_state : slai.GameState
        The tokens' state; Frozen Eye in it turns the draw pile's dedup off.
    token_kind : TokenKind
        The token kind.
    entities : list[Entity]
        Its tokens' Entities, in rank order.

    Returns
    -------
    list[int]
        One group id per Entity; [] if the token kind doesn't dedup.
    """
    if token_kind not in TOKEN_KINDS_CARD:
        return []

    # Frozen Eye shows the draw order: same-identity copies at different depths
    # are different selections
    if token_kind is TokenKind.CARD_PILE_DRAW and draw_order_visible(game_state):
        return []

    # A Card token kind's Entities are Cards
    return _card_identity_ids(cast(list[Card], entities))


def _wasted_potion_discard(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` discards a Potion while nothing needs the room.

    No Potion can arrive under Sozu, and with a free belt slot the discard only pays as a reroll
    before drinking an Entropic Brew held in another slot.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True for a Potion discard under Sozu, or with a free belt slot and no Entropic Brew in
        another slot.
    """
    if action.action_type is not ActionType.PotionDiscard:
        return False
    if any(relic.name == RelicName.Sozu for relic in game_state.relics):
        return True
    potions = game_state.potions
    return len(potions) < game_state.potion_slots_max and not any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    )


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def encode_batch_masks(
    game_states: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    batch_tokens: list[list[tuple[TokenKind, Entity]]],
    batch_row_of_id: list[dict[int, int]],
    np_token_kind: np.ndarray,
    np_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode the masks from each state's legal actions.

    Wasted Potion discards are engine-legal but masked.

    The token pass gives each state's tokens, as (token kind, Entity) in window order, and each
    Entity id's first row. A legal id's bit goes on that row, so the masks read the encoder's
    placement rather than re-deriving it, and an Entity the encoder dropped is unselectable by
    construction.

    Parameters
    ----------
    game_states : list[slai.GameState]
        The states, B of them.
    batch_legal_actions : list[list[slai.Action]]
        Each state's legal actions.
    batch_tokens : list[list[tuple[TokenKind, Entity]]]
        Each state's tokens, as (token kind, Entity) in window order.
    batch_row_of_id : list[dict[int, int]]
        Each state's Entity ids, each mapped to its first row.
    np_token_kind : np.ndarray
        (N,) int16. Each token's token kind.
    np_count : np.ndarray
        (B,) int64. Each state's number of tokens.

    Returns
    -------
    np_legal : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The tokens' legal bits, per action type, deduped by Card
        identity within each token kind.
    np_needs_target : np.ndarray
        (N,) bool. The tokens' needs-target bits: set where a legal action carries a Monster
        target.
    np_legal_action_type : np.ndarray
        (B, NUM_ACTION_TYPES) bool. The states' legal action types.
    """

    # Each state's first row and the end of the last
    np_start = np.concatenate([[0], np.cumsum(np_count)])

    # Initialize the masks: each token's legal action types and its needs-target bit, and each
    # state's legal action types
    np_legal = np.zeros((len(np_token_kind), NUM_ACTION_TYPES), dtype=np.bool_)
    np_needs_target = np.zeros(len(np_token_kind), dtype=np.bool_)
    np_legal_action_type = np.zeros((len(game_states), NUM_ACTION_TYPES), dtype=np.bool_)

    for b, game_state in enumerate(game_states):
        tokens = batch_tokens[b]
        row_of_id = batch_row_of_id[b]
        start, end = int(np_start[b]), int(np_start[b + 1])

        # The (type, token kind) pairs whose legal ids set a bit, for the dedup below
        legal_token_kinds: set[tuple[int, TokenKind]] = set()

        for action in batch_legal_actions[b]:
            if _wasted_potion_discard(game_state, action):
                continue

            at = int(action.action_type)
            ids = action.ids  # Bound once: each read clones the list
            if ids:
                row = row_of_id.get(ids[0])
                if row is None:
                    # No token (cap overflow, or the encoder disagrees with the engine):
                    # unselectable, so its type bit stays off
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(f"Dropped legal action: type={at} id {ids[0]} has no token")
                    continue

                np_legal[row, at] = True  # Per ACTION TYPE — legality differs per type
                legal_token_kinds.add((at, tokens[row - start][0]))

                # L3: only CardPlay/PotionUse carry a Monster target as ids[1]. The
                # engine offers every alive Monster to every targeted action, so the
                # targets live on the Monster tokens and the selection only records that
                # it takes one
                if len(ids) == 2:
                    np_needs_target[row] = True
            np_legal_action_type[b, at] = True  # L1: every legal ActionType the masks can express
        if not np_legal_action_type[b].any():
            # Every legal action was dropped — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(game_state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[b]]}"
            )

        # Dedup each type's legal tokens by Card identity per token kind (copies collapse)
        for at, token_kind in legal_token_kinds:
            # The token kind's tokens in rank order, as their indices in the state's tokens
            np_idx = np.flatnonzero(np_token_kind[start:end] == token_kind)
            group_ids = _pile_ids(game_state, token_kind, [tokens[i][1] for i in np_idx])
            if not group_ids:
                continue

            seen: set[int] = set()
            for rank, row in enumerate(start + np_idx):
                if not np_legal[row, at]:
                    continue
                if group_ids[rank] in seen:
                    np_legal[row, at] = False
                else:
                    seen.add(group_ids[rank])
    return np_legal, np_needs_target, np_legal_action_type
