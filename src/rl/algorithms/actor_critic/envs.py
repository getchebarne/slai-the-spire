import numpy as np
from slai import Action
from slai import ActionType
from slai import GameEnv
from slai import GameState

from src.rl.algorithms.actor_critic.stats import EpisodeStats
from src.rl.algorithms.actor_critic.stats import tally_step
from src.rl.constants import FAST_MODE
from src.rl.constants import NEOW
from src.rl.reward import compute_reward


_MAX_STEPS_EPISODE = 3000
_SCRIPTED_ACTION_TYPES = frozenset({ActionType.RewardTakeGold, ActionType.RewardTakeGoldStolen})


def new_env(ascension: int, seed: int) -> tuple[GameEnv, GameState]:
    env = GameEnv(ascension=ascension, fast_mode=FAST_MODE, neow=NEOW)
    return env, env.reset(seed=seed)


def step_env(
    env: GameEnv, game_state: GameState, stats: EpisodeStats, action: Action, gamma: float
) -> tuple[GameState, np.ndarray, bool]:
    """Play `action` in a game and tally it into its episode's stats.

    Parameters
    ----------
    env : GameEnv
        The game.
    game_state : GameState
        The game's state before the action.
    stats : EpisodeStats
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
    truncated = not terminated and stats.length + 1 >= _MAX_STEPS_EPISODE
    done = terminated or truncated

    # Compute transition reward
    reward = compute_reward(game_state, game_state_next, done, gamma, action)  # (K,)

    # Compute stats
    tally_step(stats, game_state, game_state_next, action, reward, gamma, done, truncated)

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
