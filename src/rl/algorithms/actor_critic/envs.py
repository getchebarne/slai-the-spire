from dataclasses import dataclass, field

import numpy as np
from slai import Action
from slai import ActionType
from slai import GameEnv
from slai import GameState

from src.rl.constants import FAST_MODE
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import NEOW
from src.rl.reward import RewardStream
from src.rl.reward import compute_reward
from src.rl.reward import game_won
from src.rl.reward import shop_gold_spent


_MAX_STEPS_EPISODE = 3000
_SCRIPTED_ACTION_TYPES = frozenset({ActionType.RewardTakeGold, ActionType.RewardTakeGoldStolen})


@dataclass
class EpisodeResult:
    """An episode's tally, zeroed by default; step_env fills the fields it settles at its end."""

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


def new_env(ascension: int, seed: int) -> tuple[GameEnv, GameState]:
    env = GameEnv(ascension=ascension, fast_mode=FAST_MODE, neow=NEOW)
    return env, env.reset(seed=seed)


def step_env(
    env: GameEnv, game_state: GameState, episode_result: EpisodeResult, action: Action, gamma: float
) -> tuple[GameState, np.ndarray, bool]:
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
    done : bool
        Whether the episode ended, terminated or truncated.
    """
    game_state_next, terminated = _step_scripted(env, action)

    # A game at the step limit ends there, truncated, as a death
    truncated = not terminated and episode_result.length + 1 >= _MAX_STEPS_EPISODE
    done = terminated or truncated

    # Compute transition reward
    reward = compute_reward(game_state, game_state_next, done, gamma, action)  # (K,)

    # Compute `EpisodeResult`
    _tally_step(episode_result, game_state, game_state_next, action, reward, gamma, done, truncated)

    # Return next state, transition reward and done flag
    return game_state_next, reward, done


def _step_scripted(env: GameEnv, action: Action) -> tuple[GameState, bool]:
    """Run `env.step(action)`, then the scripted actions until none is legal.

    All of them run inside this one policy step.

    Parameters
    ----------
    env : GameEnv
        The game to step.
    action : Action
        The policy's action.

    Returns
    -------
    game_state : GameState
        The state after the last action taken.
    terminated : bool
        Whether the game ended.
    """
    game_state, terminated = env.step(action)
    while not terminated:
        actions_scriptable = [a for a in env.get_legal_actions() if a.action_type in _SCRIPTED_ACTION_TYPES]
        if not actions_scriptable:
            break
        game_state, terminated = env.step(actions_scriptable[0])
    return game_state, terminated


def _tally_step(
    stats: EpisodeResult,
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
    stats : EpisodeResult
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
