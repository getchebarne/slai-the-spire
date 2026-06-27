"""Per-episode outcome stats shared by the rollout collector and the eval battery."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals
    length: int
    won: bool
    floor: int

    @property
    def total_reward(self) -> float:
        return float(self.stream_rewards.sum())


def aggregate_episodes(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Mean outcome metrics over a batch of episodes (win rate, floor, length, reward)."""
    n = len(episodes)
    return {
        "win_rate": sum(e.won for e in episodes) / n,
        "floor_avg": sum(e.floor for e in episodes) / n,
        "length_avg": sum(e.length for e in episodes) / n,
        "reward_avg": float(np.mean([e.total_reward for e in episodes])),
    }
