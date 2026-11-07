"""Feature layout and static catalogs for focus rows."""

from enum import IntEnum
from typing import get_args

from slai import CandidatePool
from slai import ChestKind
from slai import EffectBombArm
from slai import GameEnv
from slai import PendingEffect
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo

from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.schema.effect import catalog_effects
from src.rl.schema.layout import make_channel_allocator


class Focus(IntEnum):
    REWARD = 0
    COMBAT = 1
    SHOP = 2
    CHEST = 3
    REST_SITE = 4
    EVENT = 5
    MAP = 6


# The engine's halting kinds, straight from the FFI's PendingEffect union, so a new halting
# kind widens this tuple instead of raising the ValueError in _encode_focus_into
EFFECT_PENDING = get_args(PendingEffect)
EFFECT_PENDING_TO_IDX = {cls: i for i, cls in enumerate(EFFECT_PENDING)}

_NUM_CHEST_KINDS = len(ChestKind.members())


BOMB_TURNS = GameEnv.BOMB_FUSE_TURNS  # The Bomb's fuse length

# The catalog's biggest bomb; bombs due on the same turn add up and clamp to it
BOMB_DAMAGE_MAX = max(
    effect.damage for effect in catalog_effects() if isinstance(effect, EffectBombArm)
)

CHANNEL_LAYOUT_FOCUS, _alloc = make_channel_allocator()
OFF_FOCUS_KIND = _alloc("kind", len(Focus))
OFF_FOCUS_PENDING_KIND = _alloc("pending_kind", len(EFFECT_PENDING))  # All-zero = no pending halt
OFF_FOCUS_PENDING_COUNT = _alloc("pending_count")
OFF_FOCUS_PENDING_SKIPPABLE = _alloc(
    "pending_skippable"
)  # `InputUpTo` is exactly when `SelectionSkip` is legal
OFF_FOCUS_BOMB_DAMAGE_DUE = _alloc(
    "bomb_damage_due", BOMB_TURNS
)  # Per fuse turn left: the damage the bombs deal then
OFF_FOCUS_CHEST_KIND = _alloc("chest_kind", _NUM_CHEST_KINDS)
OFF_FOCUS_CHEST_OPENED = _alloc("chest_opened")
OFF_FOCUS_REST_DONE = _alloc("rest_done")
OFF_FOCUS_PRESENCE = _alloc(
    "presence", 6
)  # Context-presence bits: Combat/reward/shop/event/rest/chest
OFF_FOCUS_DRAW_ORDER_VISIBLE = _alloc(
    "draw_order_visible"
)  # Frozen Eye: the Actor may see the draw pile's order

ENCODING_DIM_FOCUS = sum(w for _, w in CHANNEL_LAYOUT_FOCUS)

# Every pool the engine can halt over; the selection count normalizes against the
# pool's size ceiling (Combat piles are bounded by the deck)
INPUT_POOL_CAP = {
    CandidatePool.CardPileHand: MAX_SIZE_CARD_PILE_HAND,
    CandidatePool.CardPileDeck: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileDiscover: MAX_SIZE_CARD_PILE_DISCOVER,
    CandidatePool.CardPileDraw: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileDiscard: MAX_SIZE_CARD_PILE_DECK,
    CandidatePool.CardPileEventRoll: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    CandidatePool.RelicEventRoll: MAX_SIZE_RELIC_EVENT_ROLL,
}


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
    if set(INPUT_POOL_CAP) != halting:
        raise RuntimeError(
            "Halting-pool tables drifted: INPUT_POOL_CAP and the catalog's halting pools "
            f"differ on {set(INPUT_POOL_CAP) ^ halting}"
        )


_check_tables()
