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
    CandidatePool.Hand: ("combat", "hand", "hand"),
    CandidatePool.PileDraw: ("combat", "pile_draw", "draw pile"),
    CandidatePool.PileDiscard: ("combat", "pile_discard", "discard pile"),
    CandidatePool.PileExhaust: ("combat", "pile_exhaust", "exhaust pile"),
    CandidatePool.Discover: ("combat", "discover", "discovery"),
    CandidatePool.Deck: ("state", "deck", "deck"),
    CandidatePool.EventRollCard: ("event", "roll_cards", "offered cards"),
    CandidatePool.EventRollRelic: ("event", "roll_relics", "staked relics"),
    CandidatePool.EventRollPotion: ("event", "roll_potions", "offered potions"),
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
