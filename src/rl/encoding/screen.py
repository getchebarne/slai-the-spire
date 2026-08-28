from enum import IntEnum
from typing import get_args

import numpy as np
import torch
from slai import CandidatePool
from slai import Chest
from slai import ChestKind
from slai import Combat
from slai import Event
from slai import GameState
from slai import PendingEffect
from slai import RestSite
from slai import Reward
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo
from slai import Shop

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.types import PENDING_POOL_SLICE


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


def _focus_from(
    reward: Reward | None,
    combat: Combat | None,
    shop: Shop | None,
    chest: Chest | None,
    rest_site: RestSite | None,
    event: Event | None,
) -> Focus:
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


# The engine's halting kinds, straight from the FFI: `pending` is typed
# PyPendingEffect, so a new halting kind widens this tuple automatically instead
# of raising the "unencoded pending halt kind" ValueError below
EFFECT_PENDING = get_args(PendingEffect)
_EFFECT_PENDING_TO_IDX = {cls: i for i, cls in enumerate(EFFECT_PENDING)}

_NUM_CHEST_KINDS = len(ChestKind.members())

# Layout offsets
_OFF_PENDING = len(Focus)
_OFF_COUNT = _OFF_PENDING + len(EFFECT_PENDING)
_OFF_SKIPPABLE = _OFF_COUNT + 1
_OFF_BOMB = _OFF_SKIPPABLE + 1
_OFF_CHEST_KIND = _OFF_BOMB + 1
_OFF_CHEST_OPENED = _OFF_CHEST_KIND + _NUM_CHEST_KINDS
_OFF_REST_DONE = _OFF_CHEST_OPENED + 1
_OFF_PRESENCE = _OFF_REST_DONE + 1
ENCODING_DIM_SCREEN = _OFF_PRESENCE + 6

_BOMB_TURNS = 3.0  # The Bomb's fuse length (engine process_effect_turn_end.rs, bare literal)

# Every pool the engine can halt over; the pick count normalizes against its slice cap
_INPUT_POOL_CAP = {
    CandidatePool.Hand: MAX_SIZE_HAND,
    CandidatePool.Deck: MAX_SIZE_DECK,
    CandidatePool.Discover: MAX_SIZE_DISCOVER,
    CandidatePool.PileDraw: MAX_SIZE_DRAW_PILE,
    CandidatePool.PileDiscard: MAX_SIZE_DISC_PILE,
    CandidatePool.PileExhaust: MAX_SIZE_EXHAUST,
    CandidatePool.EventRollCard: 1,
    CandidatePool.EventRollRelic: 2,
    CandidatePool.EventRollPotion: 1,
}
assert set(_INPUT_POOL_CAP) == set(PENDING_POOL_SLICE), "halting-pool tables drifted"


def _encode_screen_into(state: GameState, out: np.ndarray) -> None:
    # Every context/pending access deep-clones its snapshot across the FFI —
    # fetch each exactly once and derive everything from the locals
    combat = state.combat
    reward = state.reward
    shop = state.shop
    chest = state.chest
    rest_site = state.rest_site
    event = state.event
    pending = state.effect_pending

    out[_focus_from(reward, combat, shop, chest, rest_site, event)] = 1.0

    # Pending input kind OHE (all-zero = no pending) + normalized remaining pick
    # count + skippable flag (InputUpTo is exactly when PickSkip is legal)
    if pending is not None:
        idx = _EFFECT_PENDING_TO_IDX.get(type(pending))
        if idx is None:
            raise ValueError(f"Unencoded pending halt kind: {type(pending).__name__}")
        out[_OFF_PENDING + idx] = 1.0

        target = pending.target
        selection_kind = target.selection_kind if target is not None else None
        if isinstance(selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            cap = _INPUT_POOL_CAP.get(target.candidate_pool)
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {target.candidate_pool}"
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
    # over a live combat/chest/event host; Colosseum's combat over an event)
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
