import random
from dataclasses import dataclass

import numpy as np
import slai

from src.rl.algorithms.actor_critic.curriculum import env_level
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import finish_episode
from src.rl.algorithms.actor_critic.episode import make_episode_stats
from src.rl.algorithms.actor_critic.episode import tally_step
from src.rl.reward import compute_reward
from src.rl.utils import new_env
from src.rl.utils import step_scripted


@dataclass
class ShardSettings:
    """The settings of one worker's games, fixed for the run."""

    num_envs: int

    # The shard plays games [first_env, first_env + num_envs) of total_envs; the curriculum
    # level reads that global index, so the level mix doesn't depend on the sharding
    first_env: int
    total_envs: int
    gamma: float
    ascension_start: float
    ascension_max: int
    seed: int


@dataclass
class Shard:
    """One worker's games, each with its episode's running tally."""

    envs: list[slai.GameEnv]
    obs: list[slai.GameState]
    stats: list[EpisodeStats]
    ascension_c: float  # Soft-curriculum position; envs adopt their level on reset


def _make_env(ascension: int) -> tuple[slai.GameEnv, slai.GameState]:
    env = new_env(ascension)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))
    return env, obs


def make_shard(settings: ShardSettings) -> Shard:
    # Each game's tally lives in the shard's `stats`, not per rollout, so an episode that
    # spans rollouts reports its true totals
    envs, obs = [], []
    for j in range(settings.num_envs):
        level = env_level(
            settings.ascension_start,
            settings.first_env + j,
            settings.total_envs,
            settings.ascension_max,
        )
        env, ob = _make_env(level)
        envs.append(env)
        obs.append(ob)
    return Shard(
        envs=envs,
        obs=obs,
        stats=[make_episode_stats() for _ in range(settings.num_envs)],
        ascension_c=settings.ascension_start,
    )


def step_env(
    shard: Shard, j: int, action: slai.Action, settings: ShardSettings
) -> tuple[np.ndarray, EpisodeStats | None]:
    """Play `action` in game j.

    Parameters
    ----------
    shard : Shard
        The worker's games. Game j's state and tally are updated in place, and a game whose
        episode ended is replaced by a new one at its curriculum level.
    j : int
        The game's index in the shard.
    action : slai.Action
        The action to play.
    settings : ShardSettings
        The shard's settings, for the reward's discount and a new game's level.

    Returns
    -------
    reward : np.ndarray
        (K,) float64. The step's reward, one per reward stream.
    finished : EpisodeStats | None
        The episode's stats if the episode ended, None otherwise.
    """
    prev = shard.obs[j]
    nxt, terminated = step_scripted(shard.envs[j], action)
    reward = compute_reward(prev, nxt, terminated, settings.gamma, action)  # (K,)
    tally_step(shard.stats[j], prev, nxt, action, reward, settings.gamma)
    finished = None
    if terminated:
        # nxt is still the pre-reset terminal snapshot here
        finish_episode(shard.stats[j], nxt, terminated)
        finished, shard.stats[j] = shard.stats[j], make_episode_stats()
        level = env_level(
            shard.ascension_c, settings.first_env + j, settings.total_envs, settings.ascension_max
        )
        shard.envs[j], nxt = _make_env(level)
    shard.obs[j] = nxt
    return reward, finished
