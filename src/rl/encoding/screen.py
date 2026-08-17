from enum import IntEnum

import numpy as np
import torch
from slai import CandidatePoolDeck
from slai import CandidatePoolDiscover
from slai import CandidatePoolHand
from slai import CandidatePoolPileDiscard
from slai import CandidatePoolPileDraw
from slai import ChestKind
from slai import EffectBonfireOffer
from slai import EffectCardBottle
from slai import EffectCardDiscard
from slai import EffectCardDiscoverPick
from slai import EffectCardDuplicate
from slai import EffectCardExhaust
from slai import EffectCardMove
from slai import EffectCardNightmarePick
from slai import EffectCardPurge
from slai import EffectCardRetain
from slai import EffectCardSetupPick
from slai import EffectCardTransform
from slai import EffectCardUpgrade
from slai import GameState
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo
from slai import members

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_HAND


class Focus(IntEnum):
    """Trainer-side phase discriminant. The engine deleted the screen enum; the
    active phase is which nullable context is set, resolved with the engine's own
    precedence (src/utils.rs context_focus): an active Reward overlays its host
    (combat loot, opened chest), and Colosseum stacks a combat over a live event."""

    REWARD = 0
    COMBAT = 1
    SHOP = 2
    CHEST = 3
    REST_SITE = 4
    EVENT = 5
    MAP = 6


def _focus_from(reward, combat, shop, chest, rest_site, event) -> Focus:
    """Precedence over pre-fetched context snapshots (FFI accesses deep-clone —
    callers that already hold the contexts must not re-fetch them)."""
    if reward is not None:
        return Focus.REWARD
    if combat is not None:
        return Focus.COMBAT
    if shop is not None:
        return Focus.SHOP
    if chest is not None:
        return Focus.CHEST
    if rest_site is not None:
        return Focus.REST_SITE
    if event is not None:
        return Focus.EVENT
    return Focus.MAP


def derive_focus(state: GameState) -> Focus:
    return _focus_from(
        state.reward, state.combat, state.shop, state.chest, state.rest_site, state.event
    )


# The exact halting kinds of the engine's pending-effect dispatcher
# (action.rs fill_legal_actions_effect_pending); a new halting kind must be
# added here deliberately — encoding fails loud (ValueError) below.
_EFFECT_PENDING = (
    EffectCardDiscard,
    EffectCardRetain,
    EffectCardExhaust,
    EffectCardSetupPick,
    EffectCardNightmarePick,
    EffectCardMove,
    EffectCardDiscoverPick,
    EffectCardPurge,
    EffectBonfireOffer,
    EffectCardBottle,
    EffectCardUpgrade,
    EffectCardDuplicate,
    EffectCardTransform,
)
_EFFECT_PENDING_TO_IDX = {cls: i for i, cls in enumerate(_EFFECT_PENDING)}

_NUM_CHEST_KINDS = len(members(ChestKind))

# Layout offsets
_OFF_PENDING = len(Focus)
_OFF_COUNT = _OFF_PENDING + len(_EFFECT_PENDING)
_OFF_SKIPPABLE = _OFF_COUNT + 1
_OFF_BOMB = _OFF_SKIPPABLE + 1
_OFF_CHEST_KIND = _OFF_BOMB + 1
_OFF_CHEST_OPENED = _OFF_CHEST_KIND + _NUM_CHEST_KINDS
_OFF_REST_DONE = _OFF_CHEST_OPENED + 1
_OFF_PRESENCE = _OFF_REST_DONE + 1
ENCODING_DIM_SCREEN = _OFF_PRESENCE + 6

_BOMB_TURNS = 3.0  # The Bomb's fuse length (engine process_effect_turn_end.rs, bare literal)

# Pool the remaining-pick count is normalized against (Input/InputUpTo halts).
# CardMove halts carry PileDraw (Violence, Secret Weapon/Technique) or
# PileDiscard (Liquid Memories) as the source pool.
_INPUT_POOL_CAP = {
    CandidatePoolHand: MAX_SIZE_HAND,
    CandidatePoolDeck: MAX_SIZE_DECK,
    CandidatePoolDiscover: MAX_SIZE_DISCOVER,
    CandidatePoolPileDraw: MAX_SIZE_DRAW_PILE,
    CandidatePoolPileDiscard: MAX_SIZE_DISC_PILE,
}


def _encode_screen_into(state: GameState, out: np.ndarray) -> None:
    # Every context/pending access deep-clones its snapshot across the FFI —
    # fetch each exactly once and derive everything from the locals.
    combat = state.combat
    reward = state.reward
    shop = state.shop
    chest = state.chest
    rest_site = state.rest_site
    event = state.event
    pending = state.pending

    out[_focus_from(reward, combat, shop, chest, rest_site, event)] = 1.0

    # Pending input kind OHE (all-zero = no pending) + normalized remaining pick
    # count + skippable flag (InputUpTo is exactly when PickSkip is legal).
    if pending is not None:
        idx = _EFFECT_PENDING_TO_IDX.get(type(pending))
        if idx is None:
            raise ValueError(f"Unencoded pending halt kind: {type(pending).__name__}")
        out[_OFF_PENDING + idx] = 1.0

        target = pending.target
        selection_kind = target.selection_kind if target is not None else None
        if isinstance(selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            cap = _INPUT_POOL_CAP.get(type(target.candidate_pool))
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {type(target.candidate_pool).__name__}"
                    f" (pending {type(pending).__name__})"
                )

            out[_OFF_COUNT] = selection_kind.count / cap
            if isinstance(selection_kind, SelectionKindInputUpTo):
                out[_OFF_SKIPPABLE] = 1.0

    if combat is not None:
        out[_OFF_BOMB] = combat.bomb_countdown / _BOMB_TURNS

    if chest is not None:
        out[_OFF_CHEST_KIND + int(chest.kind)] = 1.0
        out[_OFF_CHEST_OPENED] = float(chest.opened)

    if rest_site is not None:
        out[_OFF_REST_DONE] = float(rest_site.done)

    # Context-presence bits: the focus OHE alone can't express overlays (reward
    # over a live combat/chest/event host; Colosseum's combat over an event).
    for i, ctx in enumerate((combat, reward, shop, event, rest_site, chest)):
        if ctx is not None:
            out[_OFF_PRESENCE + i] = 1.0


def encode_batch_screen(batch_state: list[GameState], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_state)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_SCREEN), dtype=np.float32)

    for b, state in enumerate(batch_state):
        _encode_screen_into(state, np_out[b])

    return torch.from_numpy(np_out).to(device)
