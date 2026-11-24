import numpy as np
from slai import GameState

from src.rl.encoding.layout import make_channel_allocator


_CHANNEL_LAYOUT_REWARD_META, _alloc = make_channel_allocator()
_OFF_REWARD_RELICS_EXCLUSIVE = _alloc("relics_exclusive")

ENCODING_DIM_REWARD_META = sum(w for _, w in _CHANNEL_LAYOUT_REWARD_META)


def encode_reward_meta_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write reward flags, leaving states without a reward zero.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_REWARD_META,) float32. The context row's Reward range, written in place.
    """
    reward = game_state.reward
    if reward is None:
        return
    np_out[_OFF_REWARD_RELICS_EXCLUSIVE] = float(reward.relics_exclusive)
