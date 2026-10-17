"""Per-episode outcome stats shared by the rollout collector and the eval battery."""

from dataclasses import dataclass

import numpy as np
from slai import Action
from slai import ActionType
from slai import GameEnv
from slai import GameState

from src.rl.constants import MAP_HEIGHT
from src.rl.reward import ACT_SPAN
from src.rl.reward import RewardStream
from src.rl.reward import cumulative_floor
from src.rl.reward import shop_gold_spent


# Cumulative floor of Act II's first row: reaching it ⟺ beat the Act I boss.
_ACT2_FLOOR_MIN = ACT_SPAN

# Final boss room: reaching it splits the act-2 plateau into reach x conversion
_BOSS_FLOOR_MIN = (GameEnv.ACT_FINAL - 1) * ACT_SPAN + MAP_HEIGHT


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals, undiscounted
    length: int  # Policy steps: step_scripted folds the scripted gold takes into one step
    won: bool  # Game over with health > 0
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


def make_episode_stats() -> EpisodeStats:
    """A zeroed tally: won, floor and deck_size are placeholders until finish_episode"""
    return EpisodeStats(
        stream_rewards=np.zeros(len(RewardStream)),
        length=0,
        won=False,
        floor=0,
        stream_returns=np.zeros(len(RewardStream)),
        gold_gained=0,
        gold_spent=0,
        deck_size=0,
        cards_added=0,
        purges=0,
    )


def tally_step(
    stats: EpisodeStats,
    game_state: GameState,
    game_state_next: GameState,
    action: Action,
    reward: np.ndarray,
    gamma: float,
) -> None:
    """Add one policy step: `action` took game_state to game_state_next and paid `reward`"""
    stats.stream_rewards += reward
    stats.stream_returns += gamma**stats.length * reward
    stats.length += 1
    stats.gold_gained += max(0, game_state_next.character.gold - game_state.character.gold)
    stats.gold_spent += shop_gold_spent(game_state, game_state_next, action)
    stats.cards_added += max(
        0, len(game_state_next.card_pile_deck) - len(game_state.card_pile_deck)
    )
    stats.purges += action.action_type == ActionType.ShopPurge


def finish_episode(stats: EpisodeStats, game_state: GameState, terminated: bool) -> None:
    """Fill the fields an episode settles at its end, from its last snapshot"""
    stats.won = bool(terminated and game_state.character.health > 0)
    stats.floor = cumulative_floor(game_state)
    stats.deck_size = len(game_state.card_pile_deck)


# Metric-name and prefix constants — shared by the rollout logging (train/*), the
# eval battery (eval/*), and train()'s eval print, so producers and consumers can't drift apart
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


def aggregate_episode_stats(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Outcome metrics over a batch of episodes, keyed by the KEY_* names."""
    n = len(episodes)
    rewards = [e.total_reward for e in episodes]
    return {
        KEY_WIN_RATE: sum(e.won for e in episodes) / n,
        KEY_FLOOR_AVG: sum(e.floor for e in episodes) / n,
        KEY_LENGTH_AVG: sum(e.length for e in episodes) / n,
        KEY_REWARD_AVG: float(np.mean(rewards)),
        KEY_REWARD_STD: float(np.std(rewards)),
        # The Act-I-cleared rate: the progress signal while two-act wins stay rare
        KEY_ACT2_RATE: sum(e.floor >= _ACT2_FLOOR_MIN for e in episodes) / n,
        KEY_GOLD_AVG: sum(e.gold_gained for e in episodes) / n,
        KEY_GOLD_SPENT_AVG: sum(e.gold_spent for e in episodes) / n,
        KEY_BOSS_RATE: sum(e.floor >= _BOSS_FLOOR_MIN for e in episodes) / n,
        KEY_DECK_SIZE_AVG: sum(e.deck_size for e in episodes) / n,
        KEY_CARDS_ADDED_AVG: sum(e.cards_added for e in episodes) / n,
        KEY_PURGES_AVG: sum(e.purges for e in episodes) / n,
    }


def episode_scalars(completed: list[EpisodeStats]) -> dict[str, float]:
    """The train/* scalars of the episodes an iteration's rollouts finished, or none without one"""
    if not completed:
        return {}
    n = len(completed)
    scalars = aggregate_episode_stats(completed)
    scalars["completed_count"] = n
    for k, name in enumerate(RewardStream):
        scalars[f"reward_avg_{name}"] = sum(e.stream_rewards[k] for e in completed) / n

        # Discounted per-stream returns, the optimizer's actual objective:
        # undiscounted sums shift with episode length
        scalars[f"reward_ret_{name}"] = sum(e.stream_returns[k] for e in completed) / n

    return {PREFIX_TRAIN + key: value for key, value in scalars.items()}
