from itertools import islice

import slai
from slai import ACTION_SPEC_REGISTRY

from src.rl.constants import ACTION_TYPES
from src.rl.tokens import get_monster_entities
from src.rl.tokens import iter_tokens


def engine_action(
    idx_action_type: int, idx_selection: int, idx_target: int, game_state: slai.GameState
) -> slai.Action:
    """Decode the Model's output Actions into the Game's input Action format.

    Parameters
    ----------
    idx_action_type : int
        The action kind's index in `ACTION_TYPES`.
    idx_selection : int
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
    action_type = ACTION_TYPES[idx_action_type]
    
    if ACTION_SPEC_REGISTRY[action_type].arity == (0, 0):
        return slai.Action(action_type, [])
    
    if idx_selection < 0:
        raise ValueError("An action with a selection requires a nonnegative token index")

    token = next(islice(iter_tokens(game_state), idx_selection, idx_selection + 1), None)
    if token is None:
        raise IndexError(
            f"Selection offset {idx_selection} is outside this state's token sequence"
        )
    
    ids = [token[1].id]
    if idx_target >= 0:
        monsters = get_monster_entities(game_state)
        if idx_target >= len(monsters):
            raise IndexError(
                f"Monster idx_target rank {idx_target} is outside this state's Monster tokens"
            )
    
        ids.append(monsters[idx_target].id)

    return slai.Action(action_type, ids)
