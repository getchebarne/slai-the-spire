from enum import StrEnum
from enum import auto
from typing import cast

import numpy as np
import slai

from src.rl.tokens import get_character


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


def shop_gold_spent(game_state: slai.GameState, action: slai.Action) -> int:
    """Return the gold a Shop action pays, 0 for any other action.

    It reads the price: the state after the step also holds any gold gained on the way there.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        The step's action.

    Returns
    -------
    int
        The gold spent.
    """
    if action.action_type not in _SHOP_SPEND_ACTIONS:
        return 0

    shop = cast(slai.Shop, game_state.shop)
    if action.action_type == slai.ActionType.ShopPurge:
        return shop.purge_cost
    
    # The Shop's Card, Relic or Potion
    entity = cast(slai.Card | slai.Relic | slai.Potion, game_state.entities[action.ids[0]])

    return cast(int, entity.price)


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
        and get_character(game_state).health > 0
    )


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    done: bool,
    discount: float,
    action: slai.Action,
) -> np.ndarray:
    """Compute the per-stream reward, index-aligned with RewardStream.

    The total reward is the sum over streams. Floor pay and a game's end aren't discounted by their
    position inside the step.

    Parameters
    ----------
    game_state : slai.GameState
        The state before the step.
    game_state_next : slai.GameState
        The state after the step.
    done : bool
        Whether the episode ended this step: the game ended, or the step cut it at the episode
        step limit.
    discount : float
        The step's discount, γ^k: γ is the trainer's discount, k comes from `step_scripted`.
    action : slai.Action
        The step's action, for the gold spent at a Shop.

    Returns
    -------
    np.ndarray
        (len(RewardStream),) float64. Each stream's reward.
    """

    won = game_won(game_state_next)
    character = get_character(game_state)
    character_next = get_character(game_state_next)

    # Any end but a win pays a death: a death, a cut at the step limit, a door-shut kill
    death = done and not won
    health_next = 0 if death else character_next.health

    # Potential-based, γ^k·Φ(s') − Φ(s): the episode sum telescopes to its endpoints. γ must
    # be the trainer's discount. Φ stays on at a win on purpose: it prices the HP left
    hp = discount * _potential(health_next, character_next.health_max) - _potential(
        character.health, character.health_max
    )

    # A (1 − γ^k) time cost per step makes a death total ≈ −1 at any length. Only game_over or
    # the step-limit cut ends the run: health alone never does (Fairy Potion and Lizard Tail
    # revive)
    outcome = -(1.0 - discount)
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
    gold = _WEIGHT_GOLD_SPENT * float(shop_gold_spent(game_state, action))

    return np.array([outcome, hp, floor, gold])
