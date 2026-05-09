"""Route per-env states to a primary head group.

A wrapper that has buffered a pending card-play overrides the natural
phase routing — even though the engine is still in `CombatDefault`, the
trainer needs to fire the monster-select head to pick the target.

The routing function returns a list-of-lists indexed by HeadTypePrimary,
each sub-list holding the indices into the input wrappers/views array.
"""

import slai

from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.env_wrapper import EnvWrapper


def _phase_to_htp(phase) -> HeadTypePrimary | None:
    if isinstance(phase, slai.Phase.CombatDefault):
        return HeadTypePrimary.COMBAT_DEFAULT
    if isinstance(phase, slai.Phase.CombatAwaitDiscard):
        return HeadTypePrimary.COMBAT_CARD_DISCARD
    if isinstance(phase, slai.Phase.Map):
        return HeadTypePrimary.MAP_SELECT
    if isinstance(phase, slai.Phase.CombatReward):
        return HeadTypePrimary.CARD_REWARD
    if isinstance(phase, slai.Phase.RestSite):
        return HeadTypePrimary.REST_SITE
    if isinstance(phase, slai.Phase.GameOver):
        return None  # handled by trainer (env reset)
    # Unsupported: CombatAwaitNightmare / CombatAwaitRetain / CombatAwaitSetup
    # and the relic-reward path. See migration plan "out of scope".
    raise NotImplementedError(
        f"phase {type(phase).__name__} is not routed by the trainer yet "
        f"(slai pyi may be stale; see migration plan)"
    )


def get_route_primary(wrappers: list[EnvWrapper]) -> list[list[int]]:
    """Route wrapped envs to primary head groups.

    Returns a list indexed by int(HeadTypePrimary); each entry is the list
    of input-array indices belonging to that group.

    GameOver wrappers are dropped from routing (trainer should reset before
    encoding); a routing call that encounters one raises.
    """
    route: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]

    for i, wrapper in enumerate(wrappers):
        # Buffered-target override: even though the engine is in
        # CombatDefault, the trainer must pick a target.
        if wrapper.is_awaiting_target:
            route[HeadTypePrimary.COMBAT_MONSTER_SELECT].append(i)
            continue

        htp = _phase_to_htp(wrapper.obs.phase)
        if htp is None:
            raise RuntimeError(
                f"wrapper {i} is GameOver; trainer must reset before routing"
            )
        route[htp].append(i)

    return route
