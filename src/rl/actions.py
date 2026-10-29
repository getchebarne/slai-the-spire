import slai
from slai import ACTION_SPEC_REGISTRY

from src.rl.encoding.state import get_entities_for_token_kind
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import TokenKind
from src.rl.types import X_Action


def engine_action(
    at_int: int, token_kind: int, rank: int, target: int, game_state: slai.GameState
) -> slai.Action:
    """Build a `slai.Action` from the sampled hierarchy [ActionType, token kind, rank, target].

    The token kind and rank name a token and the target the target-th Monster token; the action
    carries their Entities' ids (`get_entities_for_token_kind`).

    Parameters
    ----------
    at_int : int
        The ActionType, as its int.
    token_kind : int
        The token's token kind, as its int; -1 when the type takes no id.
    rank : int
        The token's rank among its token kind's tokens; -1 when the type takes no id.
    target : int
        The target's rank among the Monster tokens; -1 when the type takes no Monster target.
    game_state : slai.GameState
        The state the hierarchy was sampled on.

    Returns
    -------
    slai.Action
        The engine's action.
    """
    action_type = ACTION_TYPE_BY_INT[at_int]
    if ACTION_SPEC_REGISTRY[action_type].arity == (0, 0):
        return slai.Action(action_type, [])

    ids = [get_entities_for_token_kind(TokenKind(token_kind), game_state)[rank].id]
    if target >= 0:  # CardPlay / PotionUse with a Monster target
        ids.append(get_entities_for_token_kind(TokenKind.MONSTERS, game_state)[target].id)
    return slai.Action(action_type, ids)


def action_from_actiontype(x_action: X_Action, i: int, game_state: slai.GameState) -> slai.Action:
    """Decode row `i` of a batched `X_Action` into a `slai.Action`.

    Parameters
    ----------
    x_action : X_Action
        The batched actions.
    i : int
        The row to decode.
    game_state : slai.GameState
        Row `i`'s state.

    Returns
    -------
    slai.Action
        The engine's action.
    """
    return engine_action(
        int(x_action.action_type[i]),
        int(x_action.token_kind[i]),
        int(x_action.rank[i]),
        int(x_action.target[i]),
        game_state,
    )
