from enum import StrEnum
from enum import auto

import numpy as np
import slai


_WEIGHT_HEALTH = 0.1000  # Φ scale, per HP
_WEIGHT_WIN = 5.0  # Beating the Heart; a death pays −1
_WEIGHT_DOOR = 5.0  # Passing from Act 3 into Act 4
_WEIGHT_FLOOR = 0.12  # Per floor climbed
_WEIGHT_GOLD_SPENT = _WEIGHT_FLOOR / 750  # Per gold spent at a Shop

_SHOP_SPEND_ACTIONS = frozenset(
    {
        slai.ActionType.ShopBuyCard,
        slai.ActionType.ShopBuyPotion,
        slai.ActionType.ShopBuyRelic,
        slai.ActionType.ShopPurge,
    }
)


# One critic output per stream; the policy's advantage is the sum of theirs
class RewardStream(StrEnum):
    OUTCOME = auto()
    HP = auto()
    FLOOR = auto()
    GOLD = auto()


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


def game_won(game_state: slai.GameState) -> bool:
    """Return whether the game ended with the Heart beaten.

    The engine also ends a game at Act 3's last boss with the door shut, with health above 0: that
    isn't a win.

    Parameters
    ----------
    game_state : slai.GameState
        The state to read.

    Returns
    -------
    bool
        Whether the game is over, in the final act, with health above 0.
    """
    return (
        game_state.game_over
        and game_state.act == slai.GameEnv.ACT_FINAL
        and game_state.character.health > 0
    )


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    terminated: bool,
    gamma: float,
    action: slai.Action | None = None,
    truncated: bool = False,
) -> np.ndarray:
    """Compute the per-stream reward, index-aligned with RewardStream.

    The total reward is the sum over streams.

    Parameters
    ----------
    game_state : slai.GameState
        The state before the step.
    game_state_next : slai.GameState
        The state after the step.
    terminated : bool
        Whether the game ended this step.
    gamma : float
        The trainer's discount, γ.
    action : slai.Action | None, default None
        The step's action, for the gold spent at a Shop.
    truncated : bool, default False
        Whether the step cut the game at the episode step limit, which ends it as a death.

    Returns
    -------
    np.ndarray
        (len(RewardStream),) float64. Each stream's reward.
    """

    won = game_won(game_state_next)

    # Any end but a win pays a death: a death, a cut at the step limit, a door-shut kill
    death = (terminated or truncated) and not won
    health_next = 0 if death else game_state_next.character.health

    # Potential-based, γ·Φ(s') − Φ(s): the episode sum telescopes to its endpoints. γ must
    # be the trainer's discount. Φ stays on at a win on purpose: it prices the HP left
    hp = gamma * _potential(health_next, game_state_next.character.health_max) - _potential(
        game_state.character.health, game_state.character.health_max
    )

    # A (1 − γ) time cost per step makes a death total ≈ −1 at any length. Only game_over or
    # the step-limit cut ends the run: health alone never does (Fairy Potion and Lizard Tail
    # revive)
    outcome = -(1.0 - gamma)
    if won:
        outcome += _WEIGHT_WIN
    elif death:
        outcome -= 1.0

    # The door pays on the step that passes from Act 3 into Act 4
    if game_state.act < slai.GameEnv.ACT_FINAL and game_state_next.act == slai.GameEnv.ACT_FINAL:
        outcome += _WEIGHT_DOOR

    # Event form: a potential form's per-step rent would make dying early pay
    floor = _WEIGHT_FLOOR * (game_state_next.floor - game_state.floor)

    # Shop spending only: gold gains pay nothing
    gold = _WEIGHT_GOLD_SPENT * float(shop_gold_spent(game_state, game_state_next, action))

    return np.array([outcome, hp, floor, gold])
