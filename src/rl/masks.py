import warnings
from collections.abc import Sequence

import numpy as np
import slai
from slai import ACTION_SPEC_REGISTRY
from slai import ActionType
from slai import Card
from slai import EventOption
from slai import Monster
from slai import Potion
from slai import PotionName
from slai import Relic
from slai import RelicName
from slai import Room

from src.rl.encoding.card import _get_card_entities
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.event import _get_event_option_entities
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.focus import draw_order_visible
from src.rl.encoding.map_ import _get_room_entities
from src.rl.encoding.monster import _get_monster_entities
from src.rl.encoding.potion import _get_potion_entities
from src.rl.encoding.relic import _get_relic_entities
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import TOKEN_KINDS_CARD
from src.rl.types import TOKEN_KINDS_POTION
from src.rl.types import TOKEN_KINDS_RELIC
from src.rl.types import TokenKind
from src.rl.types import X_Action
from src.rl.types import X_GameState


# An entity a legal action can name
type Entity = Card | Relic | Potion | Monster | EventOption | Room

# A state's id -> token index: each entity id's token, as its token kind and rank
type TokenIndex = dict[int, tuple[TokenKind, int]]


def token_entities(
    token_kind: TokenKind, state: slai.GameState, combat: slai.Combat | None
) -> Sequence[Entity]:
    """The entities of a token kind's tokens in rank order, from the encoders' own lists, so an
    entity's id and its token share one placement. `combat` is the caller's hoisted
    `state.combat`."""
    if token_kind in TOKEN_KINDS_CARD:
        return _get_card_entities(token_kind, state, combat)
    if token_kind in TOKEN_KINDS_RELIC:
        return _get_relic_entities(token_kind, state)
    if token_kind in TOKEN_KINDS_POTION:
        return _get_potion_entities(token_kind, state)
    match token_kind:
        case TokenKind.MONSTERS:
            return _get_monster_entities(combat)
        case TokenKind.EVENT_OPTIONS:
            return _get_event_option_entities(state.event)
        case TokenKind.ROOMS:
            return _get_room_entities(state)
        case _:
            raise ValueError(f"No action names a {token_kind.name} token")


def token_index(state: slai.GameState, combat: slai.Combat | None) -> TokenIndex:
    """Each entity id's token over the window, as its token kind and rank. An id on several tokens
    names its first in window order, where each family packs its decision pools first (the
    relic event roll before the owned relics it can repeat). `combat` is the caller's
    hoisted `state.combat`."""
    index: TokenIndex = {}
    for token_kind in (
        *TOKEN_KINDS_CARD,
        *TOKEN_KINDS_RELIC,
        *TOKEN_KINDS_POTION,
        TokenKind.MONSTERS,
        TokenKind.EVENT_OPTIONS,
        TokenKind.ROOMS,
    ):
        # The selected pile and the play queue repeat the ids of cards still in their piles
        if token_kind in (TokenKind.CARD_PILE_SELECTED, TokenKind.CARD_PLAY_QUEUE):
            continue
        for rank, entity in enumerate(token_entities(token_kind, state, combat)):
            index.setdefault(entity.id, (token_kind, rank))
    return index


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup).

    Groups by (card_key, effect_key) — the card row plus the effect rows, the
    card's full model-visible identity. Two same-name cards can carry different
    effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    selections. Not keyed on the Card object: the engine's __hash__ covers the arena
    `id`, which would make every card distinct and the dedup a no-op."""
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
    state: slai.GameState, combat: slai.Combat | None, token_kind: TokenKind
) -> list[int]:
    """Identity group-ids for a card token kind's dedup, over its packed tokens; [] if the
    token kind doesn't dedup. `combat` is the caller's hoisted `state.combat`."""
    if token_kind not in TOKEN_KINDS_CARD:
        return []

    # Frozen Eye shows the draw order: same-identity copies at different depths
    # are different selections
    if token_kind is TokenKind.CARD_PILE_DRAW and draw_order_visible(state):
        return []
    return card_identity_ids(_get_card_entities(token_kind, state, combat))


def wasted_potion_discard(state: slai.GameState, action: slai.Action) -> bool:
    """True when `action` discards a Potion while nothing needs the room: no Potion can
    arrive under Sozu, and with a free belt slot the discard only pays as a reroll
    before drinking an Entropic Brew held in another slot."""
    if action.action_type is not ActionType.PotionDiscard:
        return False
    if any(relic.name == RelicName.Sozu for relic in state.relics):
        return True
    potions = state.potions
    return len(potions) < state.potion_slots_max and not any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    )


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    x_gs: X_GameState,
) -> None:
    """Fill the masks of the CPU `x_gs` from each state's legal actions: its
    `x_legal_action_type`, and its tokens' `x_legal` and `x_needs_target`. Wasted Potion discards
    are engine-legal but masked.

    A legal id's bit goes on its token, the rank-th of its token kind among the state's tokens, as
    the state's token_index ranks it; so the masks read the encoder's placement rather than
    re-deriving it, and an entity the encoder dropped is unselectable by construction."""

    # The tokens' kinds and masks as NumPy views, so the bits go straight into the tokens, with
    # each state's first row and the end of the last
    np_start = np.concatenate([[0], np.cumsum(x_gs.tokens.x_count.numpy())])
    np_token_kind = x_gs.tokens.x_token_kind.numpy()
    np_legal = x_gs.tokens.x_legal.numpy()
    np_needs_target = x_gs.tokens.x_needs_target.numpy()
    np_legal_action_type = x_gs.x_legal_action_type.numpy()

    for b, state in enumerate(states):
        combat = state.combat  # hoisted once: FFI context accesses deep-clone
        index = token_index(state, combat)

        # Each token kind's tokens: their rows in rank order
        rows_of_token_kind: dict[TokenKind, np.ndarray] = {}

        # The (type, token kind) pairs whose legal ids set a bit, for the dedup below
        legal_token_kinds: set[tuple[int, TokenKind]] = set()

        for action in batch_legal_actions[b]:
            if wasted_potion_discard(state, action):
                continue

            at = int(action.action_type)
            ids = action.ids  # bound once: each read clones the list
            if ids:
                token = index.get(ids[0])
                if token is None:
                    # No token (cap overflow, or the encoder disagrees with the engine):
                    # unselectable, so its type bit stays off
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(f"Dropped legal action: type={at} id {ids[0]} has no token")
                    continue

                # Filled on a token kind's first use: every one up front costs most of the build
                token_kind, rank = token
                if token_kind not in rows_of_token_kind:
                    start, end = np_start[b], np_start[b + 1]
                    rows_of_token_kind[token_kind] = start + np.flatnonzero(
                        np_token_kind[start:end] == token_kind
                    )
                row = rows_of_token_kind[token_kind][rank]
                np_legal[row, at] = True  # per ACTION TYPE — legality differs per type
                legal_token_kinds.add((at, token_kind))

                # L3: only CardPlay/PotionUse carry a monster target as ids[1]. The
                # engine offers every alive monster to every targeted action, so the
                # targets live on the monster tokens and the selection only records that
                # it takes one
                if len(ids) == 2:
                    np_needs_target[row] = True
            np_legal_action_type[b, at] = True  # L1: every legal ActionType the masks can express
        if not np_legal_action_type[b].any():
            # Every legal action was dropped — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[b]]}"
            )

        # Dedup each type's legal tokens by card identity per token kind (copies collapse)
        for at, token_kind in legal_token_kinds:
            group_ids = _pile_ids(state, combat, token_kind)
            if not group_ids:
                continue

            seen: set[int] = set()
            for rank, row in enumerate(rows_of_token_kind[token_kind]):
                if not np_legal[row, at]:
                    continue
                if group_ids[rank] in seen:
                    np_legal[row, at] = False
                else:
                    seen.add(group_ids[rank])


def engine_action(
    at_int: int, token_kind: int, rank: int, target: int, state: slai.GameState
) -> slai.Action:
    """Build a `slai.Action` from the sampled hierarchy [ActionType, token kind, rank, target]
    (token kind/rank = -1 when the type takes no id, target = -1 when it takes no monster
    target).

    The token kind and rank name a token and the target the target-th monster token; the
    action carries their entities' ids (token_entities)."""
    action_type = ACTION_TYPE_BY_INT[at_int]
    if ACTION_SPEC_REGISTRY[action_type].arity == (0, 0):
        return slai.Action(action_type, [])

    combat = state.combat
    ids = [token_entities(TokenKind(token_kind), state, combat)[rank].id]
    if target >= 0:  # CardPlay / PotionUse with a monster target
        ids.append(token_entities(TokenKind.MONSTERS, state, combat)[target].id)
    return slai.Action(action_type, ids)


def action_from_actiontype(x_action: X_Action, i: int, state: slai.GameState) -> slai.Action:
    """Decode row `i` of a batched `X_Action` into a `slai.Action`"""
    return engine_action(
        int(x_action.action_type[i]),
        int(x_action.token_kind[i]),
        int(x_action.rank[i]),
        int(x_action.target[i]),
        state,
    )
