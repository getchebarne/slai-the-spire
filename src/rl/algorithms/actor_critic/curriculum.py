"""The soft ascension curriculum.

Games play at the levels around a position ascension_current, which advances on the eval
battery's wins at those same levels.
"""

import math
from dataclasses import dataclass

from src.rl.types import RunState


@dataclass
class AscensionCurriculum:
    """Where ascension_current starts, where it stops, and what it takes to advance it."""

    ascension_start: float
    ascension_max: int
    promote_win_rate: float
    promote_streak: int
    curriculum_step: float


def ascensions(ascension_current: float, num_envs: int) -> list[int]:
    """Return the ascension of each of `num_envs` games under ascension_current.

    The first round(frac(ascension_current)*num_envs) games play the frontier level
    ceil(ascension_current), the rest floor(ascension_current). step_curriculum caps
    ascension_current at ascension_max, so no game passes it.

    Parameters
    ----------
    ascension_current : float
        The soft-curriculum position.
    num_envs : int
        The number of games the levels spread over.

    Returns
    -------
    list[int]
        Each game's ascension.
    """
    base = math.floor(ascension_current)
    n_frontier = round((ascension_current - base) * num_envs)
    return [base + 1] * n_frontier + [base] * (num_envs - n_frontier)


def step_curriculum(
    run_state: RunState,
    win_rate_eval: float,
    ascension_current_eval: float,
    curriculum: AscensionCurriculum,
) -> None:
    """Advance the soft-curriculum position on sustained eval wins at it.

    Batteries played at an earlier position are skipped entirely (neither count nor reset the
    streak); a below-threshold result resets it; `promote_streak` consecutive passes graduate
    another `curriculum_step` fraction of envs. The position ascension_current is capped at
    ascension_max.

    Parameters
    ----------
    run_state : RunState
        The run's progress; its ascension_current and streak are updated in place.
    win_rate_eval : float
        The battery's win rate.
    ascension_current_eval : float
        The position the battery played at.
    curriculum : AscensionCurriculum
        The win rate a pass needs, the streak that promotes, the step and the highest level.
    """
    if ascension_current_eval != run_state.ascension_current:
        return
    if win_rate_eval < curriculum.promote_win_rate:
        run_state.streak = 0
        return
    run_state.streak += 1
    if run_state.streak < curriculum.promote_streak:
        return
    run_state.ascension_current = min(
        run_state.ascension_current + curriculum.curriculum_step, float(curriculum.ascension_max)
    )
    run_state.streak = 0
