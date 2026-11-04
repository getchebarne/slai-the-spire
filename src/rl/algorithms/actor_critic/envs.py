import random
from dataclasses import dataclass

import numpy as np
import slai

from src.rl.algorithms.actor_critic.episode import MAX_STEPS_EPISODE
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import finish_episode
from src.rl.algorithms.actor_critic.episode import make_episode_stats
from src.rl.algorithms.actor_critic.episode import tally_step
from src.rl.reward import compute_reward
from src.rl.utils import new_env
from src.rl.utils import step_scripted


@dataclass
class ShardSettings:
    """The games one worker plays, and its seed, fixed for the run."""

    # Their indices among the run's games, whose ascensions spread over all of them, so the level
    # mix doesn't depend on the sharding
    games: range
    seed: int


@dataclass
class Shard:
    """One worker's games, each with its episode's running tally."""

    batch_env: list[slai.GameEnv]
    batch_game_state: list[slai.GameState]
    batch_stats: list[EpisodeStats]


def _make_env(ascension: int) -> tuple[slai.GameEnv, slai.GameState]:
    env = new_env(ascension)
    game_state = env.reset(seed=random.randint(0, 2**31 - 1))
    return env, game_state


def make_shard(batch_ascension: list[int]) -> Shard:
    # Each game's tally lives in the shard's `batch_stats`, not per rollout, so an episode that
    # spans rollouts reports its true totals
    batch_env, batch_game_state = [], []
    for ascension in batch_ascension:
        env, game_state = _make_env(ascension)
        batch_env.append(env)
        batch_game_state.append(game_state)
    return Shard(
        batch_env=batch_env,
        batch_game_state=batch_game_state,
        batch_stats=[make_episode_stats() for _ in batch_ascension],
    )


def step_env(
    shard: Shard, j: int, action: slai.Action, gamma: float, ascension: int
) -> tuple[np.ndarray, EpisodeStats | None]:
    """Play `action` in game j.

    Parameters
    ----------
    shard : Shard
        The worker's games. Game j's state and tally are updated in place, and a game whose
        episode ended or reached MAX_STEPS_EPISODE steps is replaced by a new one at
        `ascension`.
    j : int
        The game's index in the shard.
    action : slai.Action
        The action to play.
    gamma : float
        The reward's discount.
    ascension : int
        The ascension a replacing game starts at.

    Returns
    -------
    reward : np.ndarray
        (K,) float64. The step's reward, one per reward stream.
    episode : EpisodeStats | None
        The episode's stats if the episode ended, None otherwise.
    """
    game_state = shard.batch_game_state[j]
    game_state_next, terminated = step_scripted(shard.batch_env[j], action)

    # A game at the step limit ends there, truncated, as a death
    truncated = not terminated and shard.batch_stats[j].length + 1 >= MAX_STEPS_EPISODE
    reward = compute_reward(
        game_state, game_state_next, terminated, gamma, action, truncated
    )  # (K,)
    tally_step(shard.batch_stats[j], game_state, game_state_next, action, reward, gamma)
    episode = None
    if terminated or truncated:
        # game_state_next is still the pre-reset snapshot here
        finish_episode(shard.batch_stats[j], game_state_next, terminated)
        episode, shard.batch_stats[j] = shard.batch_stats[j], make_episode_stats()
        shard.batch_env[j], game_state_next = _make_env(ascension)
    shard.batch_game_state[j] = game_state_next
    return reward, episode
