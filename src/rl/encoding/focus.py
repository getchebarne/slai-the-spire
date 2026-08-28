from enum import IntEnum
from enum import StrEnum
from enum import auto
from typing import get_args

import numpy as np
import torch
from slai import CandidatePool
from slai import Chest
from slai import ChestKind
from slai import Combat
from slai import Event
from slai import GameEnv
from slai import GameState
from slai import PendingEffect
from slai import RestSite
from slai import Reward
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo
from slai import Shop

from src.rl.constants import MAX_EVENT_ROLL_CARDS
from src.rl.constants import MAX_EVENT_ROLL_POTIONS
from src.rl.constants import MAX_EVENT_ROLL_RELICS
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.types import PENDING_POOL_SLICE
from src.rl.types import make_channel_allocator


class Focus(IntEnum):
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


class FocusChannel(StrEnum):
    KIND = auto()
    PENDING_KIND = auto()  # all-zero = no pending halt
    PENDING_COUNT = auto()
    PENDING_SKIPPABLE = auto()  # InputUpTo is exactly when PickSkip is legal
    BOMB_COUNTDOWN = auto()
    CHEST_KIND = auto()
    CHEST_OPENED = auto()
    REST_DONE = auto()
    PRESENCE = auto()  # context-presence bits: combat/reward/shop/event/rest/chest


CHANNEL_LAYOUT_FOCUS, _alloc = make_channel_allocator()
OFF_FOCUS_KIND = _alloc(FocusChannel.KIND, len(Focus))
OFF_FOCUS_PENDING_KIND = _alloc(FocusChannel.PENDING_KIND, len(EFFECT_PENDING))
OFF_FOCUS_PENDING_COUNT = _alloc(FocusChannel.PENDING_COUNT)
OFF_FOCUS_PENDING_SKIPPABLE = _alloc(FocusChannel.PENDING_SKIPPABLE)
OFF_FOCUS_BOMB_COUNTDOWN = _alloc(FocusChannel.BOMB_COUNTDOWN)
OFF_FOCUS_CHEST_KIND = _alloc(FocusChannel.CHEST_KIND, _NUM_CHEST_KINDS)
OFF_FOCUS_CHEST_OPENED = _alloc(FocusChannel.CHEST_OPENED)
OFF_FOCUS_REST_DONE = _alloc(FocusChannel.REST_DONE)
OFF_FOCUS_PRESENCE = _alloc(FocusChannel.PRESENCE, 6)

ENCODING_DIM_FOCUS = sum(w for _, w in CHANNEL_LAYOUT_FOCUS)

_BOMB_TURNS = GameEnv.BOMB_FUSE_TURNS  # The Bomb's fuse length

# Every pool the engine can halt over; the pick count normalizes against its slice cap
_INPUT_POOL_CAP = {
    CandidatePool.Hand: MAX_SIZE_HAND,
    CandidatePool.Deck: MAX_SIZE_DECK,
    CandidatePool.Discover: MAX_SIZE_DISCOVER,
    CandidatePool.PileDraw: MAX_SIZE_DRAW_PILE,
    CandidatePool.PileDiscard: MAX_SIZE_DISC_PILE,
    CandidatePool.PileExhaust: MAX_SIZE_EXHAUST,
    CandidatePool.EventRollCard: MAX_EVENT_ROLL_CARDS,
    CandidatePool.EventRollRelic: MAX_EVENT_ROLL_RELICS,
    CandidatePool.EventRollPotion: MAX_EVENT_ROLL_POTIONS,
}
assert set(_INPUT_POOL_CAP) == set(PENDING_POOL_SLICE), "halting-pool tables drifted"


def _encode_focus_into(state: GameState, out: np.ndarray) -> None:
    combat = state.combat
    reward = state.reward
    shop = state.shop
    chest = state.chest
    rest_site = state.rest_site
    event = state.event
    pending = state.effect_pending

    out[OFF_FOCUS_KIND + _focus_from(reward, combat, shop, chest, rest_site, event)] = 1.0

    # Pending input
    if pending is not None:
        idx = _EFFECT_PENDING_TO_IDX.get(type(pending))
        if idx is None:
            raise ValueError(f"Unsupported pending halt kind: {type(pending).__name__}")

        out[OFF_FOCUS_PENDING_KIND + idx] = 1.0

        target = pending.target
        selection_kind = target.selection_kind if target is not None else None
        if isinstance(selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            cap = _INPUT_POOL_CAP.get(target.candidate_pool)
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {target.candidate_pool}"
                    f" (pending {type(pending).__name__})"
                )

            out[OFF_FOCUS_PENDING_COUNT] = selection_kind.count / cap
            if isinstance(selection_kind, SelectionKindInputUpTo):
                out[OFF_FOCUS_PENDING_SKIPPABLE] = 1.0

    if combat is not None:
        out[OFF_FOCUS_BOMB_COUNTDOWN] = combat.bomb_countdown / _BOMB_TURNS

    if chest is not None:
        out[OFF_FOCUS_CHEST_KIND + int(chest.kind)] = 1.0
        out[OFF_FOCUS_CHEST_OPENED] = float(chest.opened)

    if rest_site is not None:
        out[OFF_FOCUS_REST_DONE] = float(rest_site.done)

    # Context-presence bits: the focus OHE alone can't express overlays
    for i, ctx in enumerate((combat, reward, shop, event, rest_site, chest)):
        if ctx is not None:
            out[OFF_FOCUS_PRESENCE + i] = 1.0


def encode_batch_focus(batch_state: list[GameState], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_state)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_FOCUS), dtype=np.float32)

    for b, state in enumerate(batch_state):
        _encode_focus_into(state, np_out[b])

    return torch.from_numpy(np_out).to(device)
