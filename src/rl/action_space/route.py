"""Route per-env states to a primary head, mirroring the engine dispatcher.

`recompute_legal_actions` checks `effect_pending` BEFORE `screen`, so we do too:
a non-None `pending` routes to the matching pending-pick head (by the pending
effect's variant class); otherwise we route on `screen`.

Pending effect classes are keyed by `type(p).__name__` (e.g. "PyEffect_CardDiscard")
derived from the runtime `slai.Effect.*` classes — robust to the .pyi stub omitting
the deck-pick / CardDiscoverPick variants.
"""

import slai

from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import HeadTypePrimary


# screen → htp (screen is an IntEnum mirror; compare via ==, which works against
# the raw `view.screen` pyclass enum).
_SCREEN_TO_HTP = {
    int(slai.Screen.Combat): HeadTypePrimary.COMBAT,
    int(slai.Screen.Map): HeadTypePrimary.MAP,
    int(slai.Screen.RestSite): HeadTypePrimary.REST,
    int(slai.Screen.Reward): HeadTypePrimary.REWARD,
    int(slai.Screen.Shop): HeadTypePrimary.SHOP,
    int(slai.Screen.Event): HeadTypePrimary.EVENT,
    int(slai.Screen.Chest): HeadTypePrimary.CHEST,
}

# pending effect variant class-name → htp. Keys derived from the runtime classes.
_PEND_CLS_TO_HTP = {
    slai.Effect.CardDiscard: HeadTypePrimary.PEND_DISCARD,
    slai.Effect.CardRetain: HeadTypePrimary.PEND_RETAIN,
    slai.Effect.CardSetupPick: HeadTypePrimary.PEND_SETUP,
    slai.Effect.CardNightmarePick: HeadTypePrimary.PEND_NIGHTMARE,
    slai.Effect.CardDiscoverPick: HeadTypePrimary.PEND_DISCOVER,
    slai.Effect.CardPurge: HeadTypePrimary.PEND_DECK_PURGE,
    slai.Effect.CardUpgrade: HeadTypePrimary.PEND_DECK_UPGRADE,
    slai.Effect.CardDuplicate: HeadTypePrimary.PEND_DECK_DUPLICATE,
    slai.Effect.CardTransform: HeadTypePrimary.PEND_DECK_TRANSFORM,
}
_PEND_NAME_TO_HTP = {cls.__name__: htp for cls, htp in _PEND_CLS_TO_HTP.items()}


def _route_one(view: slai.GameState) -> HeadTypePrimary | None:
    if view.game_over:
        return None
    pending = view.pending
    if pending is not None:
        htp = _PEND_NAME_TO_HTP.get(type(pending).__name__)
        if htp is None:
            raise NotImplementedError(f"unrouted pending effect {type(pending).__name__}")
        return htp
    htp = _SCREEN_TO_HTP.get(int(view.screen))
    if htp is None:
        raise NotImplementedError(f"unrouted screen {view.screen!r}")
    return htp


def get_route_primary(states: list[slai.GameState]) -> list[list[int]]:
    """Route each state to its primary head group. GameOver states raise
    (caller must reset before routing)."""
    route: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]
    for i, view in enumerate(states):
        htp = _route_one(view)
        if htp is None:
            raise RuntimeError(f"state {i} is GameOver; trainer must reset before routing")
        route[htp].append(i)
    return route
