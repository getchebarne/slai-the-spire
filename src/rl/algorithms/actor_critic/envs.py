from dataclasses import dataclass, field

import numpy as np
from slai import Action
from slai import GameEnv
from slai import GameState

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import NEOW
from src.rl.reward import RewardStream
from src.rl.reward import compute_reward
from src.rl.reward import game_won
from src.rl.reward import shop_gold_spent
from src.rl.scripted import step_scripted


_MAX_STEPS_EPISODE = 3000


@dataclass
class EpisodeResult:
    """An episode's tally, zeroed by default; step_env fills the fields it settles at its end."""

    length: int = 0         # Policy steps: step_scripted folds its scripted actions into one step
    num_discounts: int = 0  # Discounts taken: each step's k in γ^k (step_scripted)
    won: bool = False       # Beat the Heart (reward.game_won)
    floor: int = 0          # GameState.floor at the end, 0..56
    act: int = 0            # The last act reached

    reward_stream_rets: np.ndarray = field(default_factory=lambda: np.zeros(len(RewardStream)))
    gold_gained: int = 0     # Per step, the gold delta plus the Shop spend, floored at 0
    gold_spent: int = 0      # Gold spent at Shops (reward.shop_gold_spent) over the episode
    deck_size: int = 0       # Deck size at episode end
    truncated: bool = False  # Cut at _MAX_STEPS_EPISODE while the game still ran

    # Keys held at the episode's end
    keys_end: int = 0

    # Share of the Heart's max HP gone when the episode ends in its Room, else 0
    heart_damage: float = 0
    heart_reached: bool = False  # Reached the Heart's Room


def new_env(ascension: int, seed: int) -> tuple[GameEnv, GameState]:
    """Start a game at `ascension` from `seed`.

    Parameters
    ----------
    ascension : int
        The game's ascension.
    seed : int
        The game's seed.

    Returns
    -------
    env : GameEnv
        The game, with the run's `NEOW` setting.
    game_state : GameState
        Its first state.
    """
    env = GameEnv(ascension=ascension, neow=NEOW)

    return env, env.reset(seed=seed)


def step_env(
    env: GameEnv, game_state: GameState, episode_result: EpisodeResult, action: Action, gamma: float
) -> tuple[GameState, np.ndarray, int, bool]:
    """Play `action` in a game and tally it into its episode's `EpisodeResult`.

    Parameters
    ----------
    env : GameEnv
        The game.
    game_state : GameState
        The game's state before the action.
    episode_result : EpisodeResult
        The episode's running tally, updated in place, and finished once the episode ends or
        reaches _MAX_STEPS_EPISODE steps.
    action : Action
        The action to play.
    gamma : float
        The reward's discount.

    Returns
    -------
    game_state_next : GameState
        The game's state after the action.
    reward : np.ndarray
        (K,) float64. The step's reward, one per reward stream.
    num_discounts : int
        The step's discounts, k in γ^k.
    done : bool
        Whether the episode ended, terminated or truncated.
    """
    game_state_next, terminated, num_discounts = step_scripted(env, action)

    # A game at the step limit ends there, truncated, as a death
    truncated = not terminated and episode_result.length + 1 >= _MAX_STEPS_EPISODE
    done = terminated or truncated

    # Compute transition reward
    reward = compute_reward(
        game_state, game_state_next, done, gamma**num_discounts, action
    )  # (K,)

    # Compute `EpisodeResult`
    _tally_step(
        episode_result,
        game_state,
        game_state_next,
        action,
        reward,
        gamma,
        num_discounts,
        done,
        truncated,
    )

    # Return the next state, the step's reward, its discounts and the done flag
    return game_state_next, reward, num_discounts, done


def _tally_step(
    episode_result: EpisodeResult,
    game_state: GameState,
    game_state_next: GameState,
    action: Action,
    np_reward: np.ndarray,
    gamma: float,
    num_discounts: int,
    done: bool,
    truncated: bool,
) -> None:
    """Add one policy step: `action` took game_state to game_state_next and paid `np_reward`.

    On the episode's last step it also sets the fields the episode settles at its end, from
    game_state_next: won, truncated, floor, act, heart_reached, heart_damage, deck_size and
    keys_end.

    Parameters
    ----------
    episode_result : EpisodeResult
        The episode's tally, updated in place.
    game_state : GameState
        The state before the step.
    game_state_next : GameState
        The state after the step.
    action : Action
        The action the policy took.
    np_reward : np.ndarray
        (K,) float64. The step's reward, one per reward stream.
    gamma : float
        The discount, for the discounted returns.
    num_discounts : int
        The step's discounts, k in γ^k.
    done : bool
        Whether the episode ended this step.
    truncated : bool
        Whether the step cut the episode at the step limit while the game still ran.
    """
    # Increment 
    episode_result.length += 1
    episode_result.reward_stream_rets += gamma**episode_result.num_discounts * np_reward
    episode_result.num_discounts += num_discounts
    
    # Gold spent
    gold_spent = shop_gold_spent(game_state, action)
    episode_result.gold_spent += gold_spent
    episode_result.gold_gained += max(
        0, game_state_next.character.gold - game_state.character.gold + gold_spent
    )

    # The last step fills the fields an episode settles at its end, from its last snapshot
    if done:
        episode_result.won = game_won(game_state_next)
        episode_result.truncated = truncated
        episode_result.floor = game_state_next.floor
        episode_result.act = game_state_next.act

        # The Heart's Room: the reach; the damage, 0 elsewhere, averages to reach x conversion
        episode_result.heart_reached = (
            game_state_next.act == GameEnv.ACT_FINAL
            and game_state_next.map.y_current == MAP_HEIGHT
        )
        episode_result.heart_damage = _heart_damage(game_state_next) if episode_result.heart_reached else 0.0

        # Final Deck size
        episode_result.deck_size = len(game_state_next.card_pile_deck)

        # Keys held at episode end
        episode_result.keys_end = (
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
