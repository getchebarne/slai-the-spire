from enum import IntEnum
from enum import StrEnum
from enum import auto
from typing import get_args

import numpy as np
import torch
from slai import CandidatePool
from slai import ChestKind
from slai import EffectBombArm
from slai import GameEnv
from slai import GameState
from slai import PendingEffect
from slai import RelicName
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo

from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.encoding.effect import catalog_effects
from src.rl.types import make_channel_allocator


class Focus(IntEnum):
    REWARD = 0
    COMBAT = 1
    SHOP = 2
    CHEST = 3
    REST_SITE = 4
    EVENT = 5
    MAP = 6


def derive_focus(game_state: GameState) -> Focus:
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


def draw_order_visible(game_state: GameState) -> bool:
    """Return whether Frozen Eye shows the draw pile's order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    bool
        True when the Character holds Frozen Eye.
    """
    return any(relic.name == RelicName.FrozenEye for relic in game_state.relics)


# The engine's halting kinds, straight from the FFI's PendingEffect union, so a new halting
# kind widens this tuple instead of raising the ValueError in _encode_focus_into
EFFECT_PENDING = get_args(PendingEffect)
_EFFECT_PENDING_TO_IDX = {cls: i for i, cls in enumerate(EFFECT_PENDING)}

_NUM_CHEST_KINDS = len(ChestKind.members())


class FocusChannel(StrEnum):
    KIND = auto()
    PENDING_KIND = auto()  # All-zero = no pending halt
    PENDING_COUNT = auto()
    PENDING_SKIPPABLE = auto()  # `InputUpTo` is exactly when `SelectionSkip` is legal
    BOMB_DAMAGE_DUE = auto()  # Per fuse turn left: the damage the bombs deal then
    CHEST_KIND = auto()
    CHEST_OPENED = auto()
    REST_DONE = auto()
    PRESENCE = auto()  # Context-presence bits: Combat/reward/shop/event/rest/chest
    DRAW_ORDER_VISIBLE = auto()  # Frozen Eye: the Actor may see the draw pile's order


_BOMB_TURNS = GameEnv.BOMB_FUSE_TURNS  # The Bomb's fuse length

# The catalog's biggest bomb; bombs due on the same turn add up and clamp to it
_BOMB_DAMAGE_MAX = max(
    effect.damage for effect in catalog_effects() if isinstance(effect, EffectBombArm)
)

CHANNEL_LAYOUT_FOCUS, _alloc = make_channel_allocator()
OFF_FOCUS_KIND = _alloc(FocusChannel.KIND, len(Focus))
OFF_FOCUS_PENDING_KIND = _alloc(FocusChannel.PENDING_KIND, len(EFFECT_PENDING))
OFF_FOCUS_PENDING_COUNT = _alloc(FocusChannel.PENDING_COUNT)
OFF_FOCUS_PENDING_SKIPPABLE = _alloc(FocusChannel.PENDING_SKIPPABLE)
OFF_FOCUS_BOMB_DAMAGE_DUE = _alloc(FocusChannel.BOMB_DAMAGE_DUE, _BOMB_TURNS)
OFF_FOCUS_CHEST_KIND = _alloc(FocusChannel.CHEST_KIND, _NUM_CHEST_KINDS)
OFF_FOCUS_CHEST_OPENED = _alloc(FocusChannel.CHEST_OPENED)
OFF_FOCUS_REST_DONE = _alloc(FocusChannel.REST_DONE)
OFF_FOCUS_PRESENCE = _alloc(FocusChannel.PRESENCE, 6)
OFF_FOCUS_DRAW_ORDER_VISIBLE = _alloc(FocusChannel.DRAW_ORDER_VISIBLE)

ENCODING_DIM_FOCUS = sum(w for _, w in CHANNEL_LAYOUT_FOCUS)

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


def _encode_focus_into(game_state: GameState, out: np.ndarray) -> None:
    combat = game_state.combat
    chest = game_state.chest
    rest_site = game_state.rest_site
    pending = game_state.effect_pending

    out[OFF_FOCUS_KIND + derive_focus(game_state)] = 1.0

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
        for turns_left, damage in combat.bombs:
            out[OFF_FOCUS_BOMB_DAMAGE_DUE + turns_left - 1] += damage
        due = out[OFF_FOCUS_BOMB_DAMAGE_DUE : OFF_FOCUS_BOMB_DAMAGE_DUE + _BOMB_TURNS]
        due[:] = np.minimum(due, _BOMB_DAMAGE_MAX) / _BOMB_DAMAGE_MAX

    if chest is not None:
        out[OFF_FOCUS_CHEST_KIND + int(chest.kind)] = 1.0
        out[OFF_FOCUS_CHEST_OPENED] = float(chest.opened)

    if rest_site is not None:
        out[OFF_FOCUS_REST_DONE] = float(rest_site.done)

    # Context-presence bits: the Focus OHE alone can't express overlays
    for i, ctx in enumerate(
        (combat, game_state.reward, game_state.shop, game_state.event, rest_site, chest)
    ):
        if ctx is not None:
            out[OFF_FOCUS_PRESENCE + i] = 1.0

    out[OFF_FOCUS_DRAW_ORDER_VISIBLE] = float(draw_order_visible(game_state))


def encode_batch_focus(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    batch_size = len(game_states)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_FOCUS), dtype=np.float32)

    for b, game_state in enumerate(game_states):
        _encode_focus_into(game_state, np_out[b])

    return torch.from_numpy(np_out).to(device)


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
