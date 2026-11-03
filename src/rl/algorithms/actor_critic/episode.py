"""Per-episode outcome stats shared by the rollout collector and the eval battery."""

from dataclasses import dataclass

import numpy as np
from slai import Action
from slai import ActionType
from slai import GameEnv
from slai import GameState

from src.rl.constants import MAP_HEIGHT
from src.rl.reward import RewardStream
from src.rl.reward import game_won
from src.rl.reward import shop_gold_spent


# Greedy eval play can loop, and a training game can stall in an endless Combat; cap episodes
# so neither wedges its worker. Keep it well above an episode's length: a truncated run counts
# as a loss for the curriculum gate
MAX_STEPS_EPISODE = 3000


@dataclass
class EpisodeStats:
    stream_rewards: np.ndarray  # (K,) per-stream episode totals, undiscounted
    length: int  # Policy steps: step_scripted folds the scripted gold takes into one step
    won: bool  # Beat the Heart (reward.game_won)
    floor: int  # GameState.floor at the end, 0..56
    act: int  # The last act reached
    boss_reached: bool  # Reached the Heart's Room
    stream_returns: np.ndarray  # (K,) per-stream DISCOUNTED returns (γ^t·r_t sums)
    gold_gained: int  # Sum of positive gold deltas over the episode
    gold_spent: int  # Gold spent at Shops (reward.shop_gold_spent) over the episode
    deck_size: int  # Deck size at episode end
    cards_added: int  # Deck-size increases over the episode
    purges: int  # ShopPurge actions over the episode
    truncated: bool  # Cut at MAX_STEPS_EPISODE while the game still ran
    keys_at_act3: int | None  # Keys held on arriving in Act 3, 0..3; None if it never got there
    heart_damage: float | None  # Share of its max HP gone, ending in its Room; None elsewhere

    @property
    def total_reward(self) -> float:
        return float(self.stream_rewards.sum())


def make_episode_stats() -> EpisodeStats:
    """Return a zeroed tally; finish_episode fills the fields an episode settles at its end.

    Returns
    -------
    EpisodeStats
        A new episode's tally.
    """
    return EpisodeStats(
        stream_rewards=np.zeros(len(RewardStream)),
        length=0,
        won=False,
        floor=0,
        act=0,
        boss_reached=False,
        stream_returns=np.zeros(len(RewardStream)),
        gold_gained=0,
        gold_spent=0,
        deck_size=0,
        cards_added=0,
        purges=0,
        truncated=False,
        keys_at_act3=None,
        heart_damage=None,
    )


def tally_step(
    stats: EpisodeStats,
    game_state: GameState,
    game_state_next: GameState,
    action: Action,
    reward: np.ndarray,
    gamma: float,
) -> None:
    """Add one policy step: `action` took game_state to game_state_next and paid `reward`.

    Parameters
    ----------
    stats : EpisodeStats
        The episode's tally, updated in place.
    game_state : GameState
        The state before the step.
    game_state_next : GameState
        The state after the step.
    action : Action
        The action the policy took.
    reward : np.ndarray
        (K,) float64. The step's reward, one per reward stream.
    gamma : float
        The discount, for the discounted returns.
    """
    stats.stream_rewards += reward
    stats.stream_returns += gamma**stats.length * reward
    stats.length += 1
    stats.gold_gained += max(0, game_state_next.character.gold - game_state.character.gold)
    stats.gold_spent += shop_gold_spent(game_state, game_state_next, action)
    stats.cards_added += max(
        0, len(game_state_next.card_pile_deck) - len(game_state.card_pile_deck)
    )
    stats.purges += action.action_type == ActionType.ShopPurge
    if game_state.act < 3 and game_state_next.act == 3:
        stats.keys_at_act3 = (
            game_state_next.key_emerald + game_state_next.key_ruby + game_state_next.key_sapphire
        )


def _heart_damage(game_state: GameState) -> float:
    """Return the share of the Heart's max HP gone.

    Parameters
    ----------
    game_state : GameState
        The episode's last snapshot, in the Heart's Room.

    Returns
    -------
    float
        The share; 1.0 with the Heart dead, as a win closes the Combat and a death in the killing
        play leaves it with no Monster.
    """
    monsters = [] if game_state.combat is None else game_state.combat.monsters
    if not monsters:
        return 1.0
    heart = monsters[0]
    return 1.0 - heart.health / heart.health_max


def finish_episode(stats: EpisodeStats, game_state: GameState, terminated: bool) -> None:
    """Fill the fields an episode settles at its end, from its last snapshot.

    Parameters
    ----------
    stats : EpisodeStats
        The episode's tally; its won, truncated, floor, act, boss_reached, heart_damage and
        deck_size are set in place.
    game_state : GameState
        The episode's last snapshot.
    terminated : bool
        Whether the game is over; an episode cut short is truncated and not won.
    """
    stats.won = game_won(game_state)
    stats.truncated = not terminated
    stats.floor = game_state.floor
    stats.act = game_state.act

    # The Heart's Room: reaching it splits the final act's plateau into reach x conversion
    stats.boss_reached = (
        game_state.act == GameEnv.ACT_FINAL and game_state.map.y_current == MAP_HEIGHT
    )
    stats.heart_damage = _heart_damage(game_state) if stats.boss_reached else None
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
KEY_ACT3_RATE = "act3_rate"
KEY_ACT4_RATE = "act4_rate"
KEY_GOLD_AVG = "gold_avg"
KEY_GOLD_SPENT_AVG = "gold_spent_avg"
KEY_BOSS_RATE = "boss_rate"
KEY_DECK_SIZE_AVG = "deck_size_avg"
KEY_CARDS_ADDED_AVG = "cards_added_avg"
KEY_PURGES_AVG = "purges_avg"
KEY_TRUNCATED_RATE = "truncated_rate"
KEY_KEYS_AT_ACT3_AVG = "keys_at_act3_avg"
KEY_HEART_DAMAGE_AVG = "heart_damage_avg"


def aggregate_episode_stats(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Compute outcome metrics over a batch of episodes.

    Parameters
    ----------
    episodes : list[EpisodeStats]
        The episodes, at least one.

    Returns
    -------
    dict[str, float]
        The metrics, keyed by the KEY_* names; the Act 3 keys' and the Heart's only when a game got
        there.
    """
    n = len(episodes)
    rewards = [e.total_reward for e in episodes]
    keys_at_act3 = [e.keys_at_act3 for e in episodes if e.keys_at_act3 is not None]
    heart_damage = [e.heart_damage for e in episodes if e.heart_damage is not None]
    metrics = {
        KEY_WIN_RATE: sum(e.won for e in episodes) / n,
        KEY_FLOOR_AVG: sum(e.floor for e in episodes) / n,
        KEY_LENGTH_AVG: sum(e.length for e in episodes) / n,
        KEY_REWARD_AVG: float(np.mean(rewards)),
        KEY_REWARD_STD: float(np.std(rewards)),
        # The Act-I-cleared, Act-II-cleared and door-passed rates: the progress signals while wins
        # stay rare
        KEY_ACT2_RATE: sum(e.act >= 2 for e in episodes) / n,
        KEY_ACT3_RATE: sum(e.act >= 3 for e in episodes) / n,
        KEY_ACT4_RATE: sum(e.act >= 4 for e in episodes) / n,
        KEY_GOLD_AVG: sum(e.gold_gained for e in episodes) / n,
        KEY_GOLD_SPENT_AVG: sum(e.gold_spent for e in episodes) / n,
        KEY_BOSS_RATE: sum(e.boss_reached for e in episodes) / n,
        KEY_DECK_SIZE_AVG: sum(e.deck_size for e in episodes) / n,
        KEY_CARDS_ADDED_AVG: sum(e.cards_added for e in episodes) / n,
        KEY_PURGES_AVG: sum(e.purges for e in episodes) / n,
        KEY_TRUNCATED_RATE: sum(e.truncated for e in episodes) / n,
    }

    # Over the games that got there
    if keys_at_act3:
        metrics[KEY_KEYS_AT_ACT3_AVG] = sum(keys_at_act3) / len(keys_at_act3)
    if heart_damage:
        metrics[KEY_HEART_DAMAGE_AVG] = sum(heart_damage) / len(heart_damage)
    return metrics


def episode_scalars(completed: list[EpisodeStats]) -> dict[str, float]:
    """Return the train/* scalars of the episodes an iteration's rollouts finished.

    Parameters
    ----------
    completed : list[EpisodeStats]
        The finished episodes, possibly none.

    Returns
    -------
    dict[str, float]
        The outcome metrics, the episode count, and each reward stream's average reward and
        discounted return, keyed by their train/* names; none without an episode.
    """
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
