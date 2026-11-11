from itertools import islice

import slai
from slai import ACTION_SPEC_REGISTRY

from src.rl.constants import ACTION_TYPE_BY_INT
from src.rl.tokens import get_entities_for_token_kind
from src.rl.tokens import iter_tokens
from src.rl.types import TokenKind


def engine_action(
    idx_action_kind: int, idx_token: int, idx_target: int, game_state: slai.GameState
) -> slai.Action:
    """Decode the Model's output Actions into the Game's input Action format.

    Parameters
    ----------
    idx_action_kind : int
        The action kind's index in `ACTION_TYPE_BY_INT`.
    idx_token : int
        The selected token's index in this state's entire token sequence, in the encoder's
        order. It remains valid when states are concatenated or reordered in a minibatch.
    idx_target : int
        The target's rank among this state's Monster tokens; -1 means the action takes no
        Monster target.
    game_state : slai.GameState
        The state the indexes refer to.

    Returns
    -------
    slai.Action
        The engine's Action, with the selected Entity's id and then the target Monster's, if
        any; with no ids when the action selects nothing.
    """
    action_kind = ACTION_TYPE_BY_INT[idx_action_kind]
    
    if ACTION_SPEC_REGISTRY[action_kind].arity == (0, 0):
        return slai.Action(action_kind, [])
    
    if idx_token < 0:
        raise ValueError("An action with a selection requires a nonnegative token index")

    token = next(islice(iter_tokens(game_state), idx_token, idx_token + 1), None)
    if token is None:
        raise IndexError(f"Token offset {idx_token} is outside this state's token sequence")
    
    ids = [token[1].id]
    if idx_target >= 0:
        monsters = get_entities_for_token_kind(TokenKind.MONSTERS, game_state)
        if idx_target >= len(monsters):
            raise IndexError(
                f"Monster idx_target rank {idx_target} is outside this state's Monster tokens"
            )
    
        ids.append(monsters[idx_target].id)

    return slai.Action(action_kind, ids)
