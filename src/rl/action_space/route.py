"""Route per-env states to a primary head group.

Routing is pure phase-based now (the EnvWrapper buffering pattern is
gone — card-targeting happens inline within COMBAT_DEFAULT). Returns a
list-of-lists indexed by HeadTypePrimary; each sub-list holds the
indices into the input states array.

`Phase.CombatReward` routes to RELIC_REWARD when `relic_rewards` is
non-empty (slai emits relic offers as part of CombatReward, not as a
separate phase). The trainer must handle the post-relic transition by
re-routing on the next step.
"""

import slai

from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import NUM_PRIMARY_HEADS


def _phase_to_htp(view: slai.GameState) -> HeadTypePrimary | None:
    phase = view.phase
    if isinstance(phase, slai.Phase.CombatDefault):
        return HeadTypePrimary.COMBAT_DEFAULT
    if isinstance(phase, slai.Phase.CombatAwaitDiscard):
        return HeadTypePrimary.COMBAT_CARD_DISCARD
    if isinstance(phase, slai.Phase.CombatAwaitRetain):
        return HeadTypePrimary.COMBAT_AWAIT_RETAIN
    if isinstance(phase, slai.Phase.CombatAwaitNightmare):
        return HeadTypePrimary.COMBAT_AWAIT_NIGHTMARE
    if isinstance(phase, slai.Phase.CombatAwaitSetup):
        return HeadTypePrimary.COMBAT_AWAIT_SETUP
    if isinstance(phase, slai.Phase.Map):
        return HeadTypePrimary.MAP_SELECT
    if isinstance(phase, slai.Phase.RestSite):
        return HeadTypePrimary.REST_SITE
    if isinstance(phase, slai.Phase.CombatReward):
        # Relic rewards take precedence — slai bundles both offer types
        # into one CombatReward halt; the trainer resolves them serially
        # (relic first, then card on the next step).
        if len(view.rewards_relic) > 0:
            return HeadTypePrimary.RELIC_REWARD
        return HeadTypePrimary.CARD_REWARD
    if isinstance(phase, slai.Phase.GameOver):
        return None
    raise NotImplementedError(
        f"unrouted phase {type(phase).__name__} (slai pyi may be ahead of trainer)"
    )


def get_route_primary(states: list[slai.GameState]) -> list[list[int]]:
    """Route each state to its primary head group.

    GameOver states are dropped from routing (caller must reset before
    re-routing); a routing call that encounters one raises.
    """
    route: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]

    for i, view in enumerate(states):
        htp = _phase_to_htp(view)
        if htp is None:
            raise RuntimeError(
                f"state {i} is GameOver; trainer must reset before routing"
            )
        route[htp].append(i)

    return route
