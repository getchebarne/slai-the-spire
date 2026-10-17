"""The soft ascension curriculum: games play at the levels around a position c, which advances
on the eval battery's wins at those same levels"""

import math
from dataclasses import dataclass


@dataclass
class AscensionCurriculum:
    """Where c starts, where it stops, and what it takes to advance it"""

    ascension_start: float
    ascension_max: int
    promote_win_rate: float
    promote_streak: int
    curriculum_step: float


def env_level(c: float, env_idx: int, num_envs: int, ascension_max: int) -> int:
    """Env -> ascension under soft-curriculum position c: the first
    round(frac(c)*num_envs) envs run the frontier level ceil(c), the rest floor(c)."""
    base = math.floor(c)
    n_frontier = round((c - base) * num_envs)
    return min(base + 1 if env_idx < n_frontier else base, ascension_max)


def frontier_level(c: float, ascension_max: int) -> int:
    """The highest level played under c: ceil(c) (== c at integers)."""
    return min(math.ceil(c), ascension_max)


def step_curriculum(
    c: float,
    streak: int,
    eval_win_rate: float,
    eval_c: float,
    curriculum: AscensionCurriculum,
) -> tuple[float, int]:
    """Advance the soft-curriculum position on sustained eval wins at it. Batteries played
    at an earlier position are skipped entirely (neither count nor reset the streak); a
    below-threshold result resets it; `promote_streak` consecutive passes graduate another
    `curriculum_step` fraction of envs. c is capped at ascension_max."""
    if eval_c != c:
        return c, streak
    if eval_win_rate < curriculum.promote_win_rate:
        return c, 0
    streak += 1
    if streak < curriculum.promote_streak:
        return c, streak
    return min(c + curriculum.curriculum_step, float(curriculum.ascension_max)), 0
