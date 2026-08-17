"""Per-episode outcome stats shared by the rollout collector and the eval battery."""

from dataclasses import dataclass

import numpy as np

from src.rl.constants import MAP_HEIGHT


# Cumulative floor of Act II's first row: reaching it ⟺ beat the Act I boss.
ACT2_FLOOR_MIN = MAP_HEIGHT + 1


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals
    length: int
    won: bool
    floor: int  # run-CUMULATIVE floor (reward.cumulative_floor), 0..31

    @property
    def total_reward(self) -> float:
        return float(self.stream_rewards.sum())


# Metric-name and prefix constants — shared by the rollout logging (train/*), the
# eval battery (eval/*), and train()'s eval print, so producers and consumers can't
# drift apart (a name drift crashed NEWERA-XV at its first eval report).
PREFIX_TRAIN = "train/"
PREFIX_EVAL = "eval/"
KEY_WIN_RATE = "win_rate"
KEY_FLOOR_AVG = "floor_avg"
KEY_LENGTH_AVG = "length_avg"
KEY_REWARD_AVG = "reward_avg"
KEY_REWARD_STD = "reward_std"
KEY_ACT2_RATE = "act2_rate"


def aggregate_episodes(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Outcome metrics over a batch of episodes (win rate, floor, length, reward)."""
    n = len(episodes)
    rewards = [e.total_reward for e in episodes]
    return {
        KEY_WIN_RATE: sum(e.won for e in episodes) / n,
        KEY_FLOOR_AVG: sum(e.floor for e in episodes) / n,
        KEY_LENGTH_AVG: sum(e.length for e in episodes) / n,
        KEY_REWARD_AVG: float(np.mean(rewards)),
        KEY_REWARD_STD: float(np.std(rewards)),
        # Wins are two-act by definition now; this is the Act-I-cleared rate —
        # the progress signal through the expected long low-win-rate phase.
        KEY_ACT2_RATE: sum(e.floor >= ACT2_FLOOR_MIN for e in episodes) / n,
    }
