from itertools import islice

import slai
from slai import ACTION_SPEC_REGISTRY

from src.rl.encoding.state import get_entities_for_token_kind
from src.rl.encoding.state import iter_tokens
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import PolicyOutput
from src.rl.types import TokenKind


def engine_action(
    action_type: int, token_offset: int, target: int, game_state: slai.GameState
) -> slai.Action:
    """Decode [action type, state-local token offset, Monster target rank].

    The token offset indexes this state's entire token sequence, using the encoder's order.
    It remains valid when states are concatenated or reordered in a minibatch. The target is
    the rank among this state's Monster tokens; -1 means the action takes no Monster target.
    """
    action_kind = ACTION_TYPE_BY_INT[action_type]
    if ACTION_SPEC_REGISTRY[action_kind].arity == (0, 0):
        return slai.Action(action_kind, [])
    if token_offset < 0:
        raise ValueError("An action with a selection requires a nonnegative token offset")

    token = next(islice(iter_tokens(game_state), token_offset, token_offset + 1), None)
    if token is None:
        raise IndexError(f"Token offset {token_offset} is outside this state's token sequence")
    ids = [token[1].id]
    if target >= 0:
        monsters = get_entities_for_token_kind(TokenKind.MONSTERS, game_state)
        if target >= len(monsters):
            raise IndexError(
                f"Monster target rank {target} is outside this state's Monster tokens"
            )
        ids.append(monsters[target].id)
    return slai.Action(action_kind, ids)


def action_from_actiontype(
    output: PolicyOutput, i: int, game_state: slai.GameState
) -> slai.Action:
    """Decode action row `i` of a PolicyOutput into a `slai.Action`.

    Parameters
    ----------
    output : PolicyOutput
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
    return engine_action(*output.actions[i].tolist(), game_state)
