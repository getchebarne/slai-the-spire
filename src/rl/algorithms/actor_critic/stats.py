from dataclasses import dataclass, field

import numpy as np
from slai import Action
from slai import GameEnv
from slai import GameState

from src.rl.constants import MAP_HEIGHT
from src.rl.reward import RewardStream
from src.rl.reward import game_won
from src.rl.reward import shop_gold_spent


@dataclass
class EpisodeStats:
    """An episode's tally, zeroed by default; tally_step fills the fields it settles at its end."""

    length: int = 0  # Policy steps: step_scripted folds the scripted gold takes into one step
    won: bool = False  # Beat the Heart (reward.game_won)
    floor: int = 0  # GameState.floor at the end, 0..56
    act: int = 0  # The last act reached
    boss_reached: bool = False  # Reached the Heart's Room

    # (K,) per-stream DISCOUNTED returns (γ^t·r_t sums)
    stream_returns: np.ndarray = field(default_factory=lambda: np.zeros(len(RewardStream)))
    gold_gained: int = 0  # Sum of positive gold deltas over the episode
    gold_spent: int = 0  # Gold spent at Shops (reward.shop_gold_spent) over the episode
    deck_size: int = 0  # Deck size at episode end
    truncated: bool = False  # Cut at _MAX_STEPS_EPISODE while the game still ran

    # Keys held at death, a cut at the step limit included; `None` after a win
    keys_at_death: int | None = None

    # Share of its max HP gone, ending in its Room; None elsewhere
    heart_damage: float | None = None


def tally_step(
    stats: EpisodeStats,
    game_state: GameState,
    game_state_next: GameState,
    action: Action,
    reward: np.ndarray,
    gamma: float,
    done: bool,
    truncated: bool,
) -> None:
    """Add one policy step: `action` took game_state to game_state_next and paid `reward`.

    On the episode's last step it also sets the fields the episode settles at its end, from
    game_state_next: won, truncated, floor, act, boss_reached, heart_damage and deck_size.

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
    done : bool
        Whether the episode ended this step.
    truncated : bool
        Whether the step cut the episode at the step limit while the game still ran.
    """
    stats.stream_returns += gamma**stats.length * reward
    stats.length += 1
    stats.gold_gained += max(0, game_state_next.character.gold - game_state.character.gold)
    stats.gold_spent += shop_gold_spent(game_state, game_state_next, action)

    # The last step fills the fields an episode settles at its end, from its last snapshot
    if done:
        stats.won = game_won(game_state_next)
        stats.truncated = truncated
        stats.floor = game_state_next.floor
        stats.act = game_state_next.act

        # The Heart's Room: reaching it splits the final act's plateau into reach x conversion
        stats.boss_reached = (
            game_state_next.act == GameEnv.ACT_FINAL
            and game_state_next.map.y_current == MAP_HEIGHT
        )
        stats.heart_damage = _heart_damage(game_state_next) if stats.boss_reached else None
        stats.deck_size = len(game_state_next.card_pile_deck)
        if not stats.won:
            stats.keys_at_death = (
                game_state_next.key_emerald
                + game_state_next.key_ruby
                + game_state_next.key_sapphire
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


# Metric-name and prefix constants — shared by the rollout logging (train/*), the
# eval battery (eval/*), and train()'s eval print, so producers and consumers can't drift apart
PREFIX_TRAIN = "train/"
PREFIX_EVAL = "eval/"
KEY_WIN_RATE = "win_rate"
KEY_FLOOR_AVG = "floor_avg"
KEY_LENGTH_AVG = "length_avg"
KEY_RETURN_AVG = "return_avg"
KEY_RETURN_STD = "return_std"
KEY_ACT2_RATE = "act2_rate"
KEY_ACT3_RATE = "act3_rate"
KEY_ACT4_RATE = "act4_rate"
KEY_GOLD_AVG = "gold_avg"
KEY_GOLD_SPENT_AVG = "gold_spent_avg"
KEY_BOSS_RATE = "boss_rate"
KEY_DECK_SIZE_AVG = "deck_size_avg"
KEY_TRUNCATED_RATE = "truncated_rate"
KEY_KEYS_AT_DEATH_AVG = "keys_at_death_avg"
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
        The metrics, keyed by the KEY_* names; the keys' only when a game died, and the Heart's only
        when a game got there.
    """
    n = len(episodes)
    returns = [float(episode.stream_returns.sum()) for episode in episodes]
    keys_at_death = [
        episode.keys_at_death for episode in episodes if episode.keys_at_death is not None
    ]
    heart_damage = [
        episode.heart_damage for episode in episodes if episode.heart_damage is not None
    ]
    metrics = {
        KEY_WIN_RATE: sum(episode.won for episode in episodes) / n,
        KEY_FLOOR_AVG: sum(episode.floor for episode in episodes) / n,
        KEY_LENGTH_AVG: sum(episode.length for episode in episodes) / n,
        KEY_RETURN_AVG: float(np.mean(returns)),
        KEY_RETURN_STD: float(np.std(returns)),
        # The Act-I-cleared, Act-II-cleared and door-passed rates: the progress signals while wins
        # stay rare
        KEY_ACT2_RATE: sum(episode.act >= 2 for episode in episodes) / n,
        KEY_ACT3_RATE: sum(episode.act >= 3 for episode in episodes) / n,
        KEY_ACT4_RATE: sum(episode.act >= 4 for episode in episodes) / n,
        KEY_GOLD_AVG: sum(episode.gold_gained for episode in episodes) / n,
        KEY_GOLD_SPENT_AVG: sum(episode.gold_spent for episode in episodes) / n,
        KEY_BOSS_RATE: sum(episode.boss_reached for episode in episodes) / n,
        KEY_DECK_SIZE_AVG: sum(episode.deck_size for episode in episodes) / n,
        KEY_TRUNCATED_RATE: sum(episode.truncated for episode in episodes) / n,
    }

    # Over the games that died, and over those that got to the Heart
    if keys_at_death:
        metrics[KEY_KEYS_AT_DEATH_AVG] = sum(keys_at_death) / len(keys_at_death)
    if heart_damage:
        metrics[KEY_HEART_DAMAGE_AVG] = sum(heart_damage) / len(heart_damage)
    return metrics


def episode_scalars(episodes: list[EpisodeStats]) -> dict[str, float]:
    """Return the train/* scalars of the episodes an iteration's rollouts finished.

    Parameters
    ----------
    episodes : list[EpisodeStats]
        The finished episodes, possibly none.

    Returns
    -------
    dict[str, float]
        The outcome metrics, the episode count, and each reward stream's discounted return, keyed
        by their train/* names; none without an episode.
    """
    if not episodes:
        return {}
    n = len(episodes)
    scalars = aggregate_episode_stats(episodes)
    scalars["completed_count"] = n
    for k, name in enumerate(RewardStream):
        # Discounted per-stream returns, the optimizer's actual objective:
        # undiscounted sums shift with episode length
        scalars[f"return_avg_{name}"] = sum(episode.stream_returns[k] for episode in episodes) / n

    return {PREFIX_TRAIN + key: value for key, value in scalars.items()}
