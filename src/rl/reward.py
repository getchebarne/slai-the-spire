from enum import StrEnum
from enum import auto

import numpy as np
import slai

from src.rl.constants import MAP_HEIGHT


_WEIGHT_HEALTH = 0.1000  # Φ scale, per HP
_WEIGHT_WIN = 3.0  # A death pays −1
_WEIGHT_FLOOR = 0.12  # Per floor climbed; capped by the win margin at γ = 0.999
_WEIGHT_GOLD_SPENT = _WEIGHT_FLOOR / 750  # Per gold spent at a Shop

_SHOP_SPEND_ACTIONS = frozenset(
    {
        slai.ActionType.ShopBuyCard,
        slai.ActionType.ShopBuyPotion,
        slai.ActionType.ShopBuyRelic,
        slai.ActionType.ShopPurge,
    }
)

ACT_SPAN = MAP_HEIGHT + 1  # Rows 0..14 + the boss-room sentinel (y_current = 15)


# One critic output per stream; the policy's advantage is the sum of theirs
class RewardStream(StrEnum):
    OUTCOME = auto()
    HP = auto()
    FLOOR = auto()
    GOLD = auto()


def cumulative_floor(game_state: slai.GameState) -> int:
    """Return the run-cumulative floor, 0..31 over two acts: (act−1)·16 + (y_current or 0).

    `y_current` is None before an act's first Room, so the act seam pays one floor.

    Parameters
    ----------
    game_state : slai.GameState
        The game state.

    Returns
    -------
    int
        The run-cumulative floor.
    """
    return (game_state.act - 1) * ACT_SPAN + (game_state.map.y_current or 0)


def _potential(health: float, health_max: float) -> float:
    """Return the health potential Φ(h) = W·(2h − h²/h_max).

    An HP costs ~2W near death and ~0 near full HP; Φ(0) = 0 and Φ(h_max) = W·h_max.

    Parameters
    ----------
    health : float
        The Character's health, h.
    health_max : float
        The Character's max health, h_max.

    Returns
    -------
    float
        The potential, Φ(h).
    """
    return _WEIGHT_HEALTH * (2.0 * health - health * health / health_max)


def shop_gold_spent(
    game_state: slai.GameState, game_state_next: slai.GameState, action: slai.Action | None
) -> int:
    """Return the gold spent by a Shop action this step (0 for any other gold decrease).

    Parameters
    ----------
    game_state : slai.GameState
        The state before the step.
    game_state_next : slai.GameState
        The state after the step.
    action : slai.Action | None
        The step's action; None spends nothing.

    Returns
    -------
    int
        The gold spent.
    """
    if action is None or action.action_type not in _SHOP_SPEND_ACTIONS:
        return 0
    return max(0, game_state.character.gold - game_state_next.character.gold)


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    game_over_flag: bool,
    gamma: float,
    action: slai.Action | None = None,
) -> np.ndarray:
    """Compute the per-stream reward, index-aligned with RewardStream.

    The total reward is the sum over streams.

    Parameters
    ----------
    game_state : slai.GameState
        The state before the step.
    game_state_next : slai.GameState
        The state after the step.
    game_over_flag : bool
        Whether the game ended this step.
    gamma : float
        The trainer's discount, γ.
    action : slai.Action | None, default None
        The step's action, for the gold spent at a Shop.

    Returns
    -------
    np.ndarray
        (len(RewardStream),) float64. Each stream's reward.
    """

    # Potential-based, γ·Φ(s') − Φ(s): the episode sum telescopes to its endpoints. γ must
    # be the trainer's discount. Φ stays on at a win on purpose: it prices the HP left
    hp = gamma * _potential(
        game_state_next.character.health, game_state_next.character.health_max
    ) - _potential(game_state.character.health, game_state.character.health_max)

    # A (1 − γ) time cost per step makes a death total ≈ −1 at any length. Only game_over
    # ends the run: health alone never does (Fairy Potion and Lizard Tail revive)
    outcome = -(1.0 - gamma)
    if game_over_flag:
        outcome += -1.0 if game_state_next.character.health <= 0 else _WEIGHT_WIN

    # Event form: a potential form's per-step rent would make dying early pay
    floor = _WEIGHT_FLOOR * float(cumulative_floor(game_state_next) - cumulative_floor(game_state))

    # Shop spending only: gold gains pay nothing
    gold = _WEIGHT_GOLD_SPENT * float(shop_gold_spent(game_state, game_state_next, action))

    return np.array([outcome, hp, floor, gold])
