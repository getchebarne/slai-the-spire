from enum import IntEnum

import numpy as np
from slai import CandidatePool
from slai import ChestKind
from slai import EffectBombArm
from slai import GameEnv
from slai import GameState
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo

from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.encoding.effect import catalog_effects
from src.rl.encoding.layout import make_channel_allocator


class Focus(IntEnum):
    REWARD = 0
    COMBAT = 1
    SHOP = 2
    CHEST = 3
    REST_SITE = 4
    EVENT = 5
    MAP = 6


def derive_focus(game_state: GameState) -> Focus:
    """Return the context the state's decision belongs to.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    Focus
        The first open context in priority order (Reward, Combat, Shop, Chest, Rest Site, Event),
        else MAP.
    """
    if game_state.reward is not None:
        return Focus.REWARD
    if game_state.combat is not None:
        return Focus.COMBAT
    if game_state.shop is not None:
        return Focus.SHOP
    if game_state.chest is not None:
        return Focus.CHEST
    if game_state.rest_site is not None:
        return Focus.REST_SITE
    if game_state.event is not None:
        return Focus.EVENT

    return Focus.MAP


_NUM_CHEST_KINDS = len(ChestKind.members())


_BOMB_TURNS = GameEnv.BOMB_FUSE_TURNS  # The Bomb's fuse length

# The catalog's biggest bomb; bombs due on the same turn add up and clamp to it
_BOMB_DAMAGE_MAX = max(
    effect.damage for effect in catalog_effects() if isinstance(effect, EffectBombArm)
)

_CHANNEL_LAYOUT_FOCUS, _alloc = make_channel_allocator()
_OFF_FOCUS_KIND = _alloc("kind", len(Focus))
_OFF_FOCUS_PENDING_COUNT = _alloc("pending_count")
_OFF_FOCUS_PENDING_SKIPPABLE = _alloc("pending_skippable")
_OFF_FOCUS_BOMB_DAMAGE_DUE = _alloc("bomb_damage_due", _BOMB_TURNS)
_OFF_FOCUS_CHEST_KIND = _alloc("chest_kind", _NUM_CHEST_KINDS)
_OFF_FOCUS_CHEST_OPENED = _alloc("chest_opened")
_OFF_FOCUS_REST_CONSUMED = _alloc("rest_consumed")
_OFF_FOCUS_PRESENCE = _alloc("presence", 6)

ENCODING_DIM_FOCUS = sum(w for _, w in _CHANNEL_LAYOUT_FOCUS)

# Every pool the engine can halt over; the selection count normalizes against the
# pool's size ceiling (Combat piles are bounded by the deck)
_INPUT_POOL_CAP = {
    CandidatePool.CardPileHand: MAX_SIZE_CARD_PILE_HAND,
    CandidatePool.CardPileDeck: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileDiscover: MAX_SIZE_CARD_PILE_DISCOVER,
    CandidatePool.CardPileDraw: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileDiscard: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileEventRoll: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    CandidatePool.RelicEventRoll: MAX_SIZE_RELIC_EVENT_ROLL,
}


def encode_focus_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write the Focus, the pending selection, Bombs, the Chest and Rest Site, and open contexts.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_FOCUS,) float32. The context row's Focus range, written in place.
    """
    combat = game_state.combat
    chest = game_state.chest
    rest_site = game_state.rest_site
    pending = game_state.effect_pending

    np_out[_OFF_FOCUS_KIND + derive_focus(game_state)] = 1.0

    # Pending input
    # None when nothing is pending or the pending Effect has no `target` field
    target = getattr(pending, "target", None)
    if target is not None:
        selection_kind = target.selection_kind
        if isinstance(selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            cap = _INPUT_POOL_CAP.get(target.candidate_pool)
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {target.candidate_pool}"
                    f" (pending {type(pending).__name__})"
                )

            np_out[_OFF_FOCUS_PENDING_COUNT] = selection_kind.count / cap
            if isinstance(selection_kind, SelectionKindInputUpTo):
                np_out[_OFF_FOCUS_PENDING_SKIPPABLE] = 1.0

    if combat is not None:
        for turns_left, damage, _ in combat.bombs:
            np_out[_OFF_FOCUS_BOMB_DAMAGE_DUE + turns_left - 1] += damage
        due = np_out[_OFF_FOCUS_BOMB_DAMAGE_DUE : _OFF_FOCUS_BOMB_DAMAGE_DUE + _BOMB_TURNS]
        due[:] = np.minimum(due, _BOMB_DAMAGE_MAX) / _BOMB_DAMAGE_MAX

    if chest is not None:
        np_out[_OFF_FOCUS_CHEST_KIND + int(chest.chest_kind)] = 1.0
        np_out[_OFF_FOCUS_CHEST_OPENED] = float(chest.chest_opened)

    if rest_site is not None:
        np_out[_OFF_FOCUS_REST_CONSUMED] = float(rest_site.consumed)

    # Context-presence bits: the Focus OHE alone can't express overlays
    for idx_context, context in enumerate(
        (combat, game_state.reward, game_state.shop, game_state.event, rest_site, chest)
    ):
        if context is not None:
            np_out[_OFF_FOCUS_PRESENCE + idx_context] = 1.0


def _check_tables() -> None:
    """Check the selection-count caps cover exactly the pools the catalog's halts select from."""
    halting: set[CandidatePool] = set()
    for effect in catalog_effects():
        # Rows the engine never targets carry no `target` field at all
        target = getattr(effect, "target", None)
        if target is None:
            continue
        if isinstance(target.selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            halting.add(target.candidate_pool)
    if set(_INPUT_POOL_CAP) != halting:
        raise RuntimeError(
            "Halting-pool tables drifted: _INPUT_POOL_CAP and the catalog's halting pools "
            f"differ on {set(_INPUT_POOL_CAP) ^ halting}"
        )


_check_tables()
