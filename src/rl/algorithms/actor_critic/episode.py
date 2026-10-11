"""Per-episode outcome stats shared by the rollout collector and the eval battery."""

from dataclasses import dataclass

import numpy as np

from src.rl.constants import MAP_HEIGHT


# Cumulative floor of Act II's first row: reaching it ⟺ beat the Act I boss.
ACT2_FLOOR_MIN = MAP_HEIGHT + 1

# Final boss room: reaching it splits the act-2 plateau into reach x conversion
BOSS_FLOOR_MIN = 2 * ACT2_FLOOR_MIN - 1


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals, undiscounted
    length: int
    won: bool
    floor: int  # run-CUMULATIVE floor (reward.cumulative_floor), 0..31
    stream_returns: np.ndarray  # (K,) per-stream DISCOUNTED returns (γ^t·r_t sums)
    gold_gained: int  # sum of positive gold deltas over the episode
    gold_spent: int  # gold spent at shops (reward.shop_gold_spent) over the episode
    deck_size: int  # deck size at episode end
    cards_added: int  # deck-size increases over the episode
    purges: int  # ShopPurge actions over the episode

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
KEY_GOLD_AVG = "gold_avg"
KEY_GOLD_SPENT_AVG = "gold_spent_avg"
KEY_BOSS_RATE = "boss_rate"
KEY_DECK_SIZE_AVG = "deck_size_avg"
KEY_CARDS_ADDED_AVG = "cards_added_avg"
KEY_PURGES_AVG = "purges_avg"


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
        KEY_GOLD_AVG: sum(e.gold_gained for e in episodes) / n,
        KEY_GOLD_SPENT_AVG: sum(e.gold_spent for e in episodes) / n,
        KEY_BOSS_RATE: sum(e.floor >= BOSS_FLOOR_MIN for e in episodes) / n,
        KEY_DECK_SIZE_AVG: sum(e.deck_size for e in episodes) / n,
        KEY_CARDS_ADDED_AVG: sum(e.cards_added for e in episodes) / n,
        KEY_PURGES_AVG: sum(e.purges for e in episodes) / n,
    }
