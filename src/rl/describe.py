"""Human-readable descriptions of engine state, shared by the CLI and the LLM
renderer.

Kept out of `test_agent.py` so `src/llm/render.py` can format a pending pick
without importing the training stack. Depends on `slai` ONLY for that reason —
anything from `src.rl.types` or `src.rl.encoding` pulls torch (~1 s). The two
tables below are cross-checked against their sources in `test_agent`, which
already pays that cost.
"""

import slai
from slai import CandidatePool


def variant_name(value: object) -> str:
    """`ModifierKind.Strength` -> `Strength` (slai pyclass enums repr that way)."""
    s = repr(value)
    return s.rsplit(".", 1)[-1] if "." in s else s


# The engine resolves every halt with one EffectPendingResolve{idx}; idx indexes
# the collection named by pending.target.candidate_pool, so rendering has to read
# the pool rather than the action type
PENDING_POOL_SOURCE: dict[CandidatePool, tuple[str, str, str]] = {
    CandidatePool.CardPileHand: ("combat", "card_pile_hand", "hand"),
    CandidatePool.CardPileDraw: ("combat", "card_pile_draw", "draw pile"),
    CandidatePool.CardPileDiscard: ("combat", "card_pile_discard", "discard pile"),
    CandidatePool.CardPileExhaust: ("combat", "card_pile_exhaust", "exhaust pile"),
    CandidatePool.CardPileDiscover: ("combat", "card_pile_discover", "discovery"),
    CandidatePool.CardPileDeck: ("state", "card_pile_deck", "deck"),
    CandidatePool.CardPileEventRoll: ("event", "card_pile_event_roll", "offered cards"),
    CandidatePool.RelicEventRoll: ("event", "relic_event_roll", "staked relics"),
    CandidatePool.PotionEventRoll: ("event", "potion_event_roll", "offered potions"),
}

PENDING_VERB: dict[type, str] = {
    slai.EffectCardDiscard: "discarded",
    slai.EffectCardRetain: "retained",
    slai.EffectCardExhaust: "exhausted",
    slai.EffectCardSetupPick: "set aside",
    slai.EffectCardNightmarePick: "copied",
    slai.EffectCardMove: "moved to hand",
    slai.EffectCardDiscoverPick: "discovered",
    slai.EffectCardPurge: "purged",
    slai.EffectBonfireOffer: "offered to the bonfire",
    slai.EffectCardBottle: "bottled",
    slai.EffectCardUpgrade: "upgraded",
    slai.EffectCardDuplicate: "duplicated",
    slai.EffectCardTransform: "transformed",
    slai.EffectRelicLose: "gave up",
    slai.EffectMatchFlipSeen: "flipped",
    slai.EffectRestSmith: "upgraded",
    slai.EffectRestToke: "removed",
}


def describe_pending_pick(view: slai.GameState, idx: int) -> str:
    """Describe an EffectPendingResolve against the pre-step pending halt."""
    pending = view.effect_pending
    if pending is None or pending.target is None:
        return f"resolved a pending pick (#{idx})"
    verb = PENDING_VERB.get(type(pending), type(pending).__name__)
    entry = PENDING_POOL_SOURCE.get(pending.target.candidate_pool)
    if entry is None:
        return f"{verb} #{idx}"
    owner, attr, noun = entry
    source = {"combat": view.combat, "state": view, "event": view.event}[owner]
    pile = getattr(source, attr) if source is not None else []
    what = variant_name(pile[idx].name) if idx < len(pile) else f"#{idx}"
    return f"{verb} {what} (from {noun})"
