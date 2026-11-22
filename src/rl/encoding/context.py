"""Dense per-state features in the global projection's established column order."""

import numpy as np
from slai import GameState

from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.energy import encode_energy_into
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.event import encode_event_meta_into
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import encode_focus_into
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import encode_map_meta_into
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.reward import encode_reward_meta_into
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.shop import encode_shop_meta_into


_ENCODING_DIM_ASCENSION = ASCENSION_LEVEL

_ENERGY = slice(0, ENCODING_DIM_ENERGY)
_FOCUS = slice(_ENERGY.stop, _ENERGY.stop + ENCODING_DIM_FOCUS)
_ASCENSION = slice(_FOCUS.stop, _FOCUS.stop + _ENCODING_DIM_ASCENSION)
_MAP_META = slice(_ASCENSION.stop, _ASCENSION.stop + ENCODING_DIM_MAP_META)
_REWARD_META = slice(_MAP_META.stop, _MAP_META.stop + ENCODING_DIM_REWARD_META)
_SHOP_META = slice(_REWARD_META.stop, _REWARD_META.stop + ENCODING_DIM_SHOP_META)
_EVENT_META = slice(_SHOP_META.stop, _SHOP_META.stop + ENCODING_DIM_EVENT_META)
ENCODING_DIM_CONTEXT = _EVENT_META.stop


def encode_context_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write one state's dense features into its initially zeroed context row.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_CONTEXT,) float32. The state's context row, written in place.
    """
    encode_energy_into(game_state, np_out[_ENERGY])
    encode_focus_into(game_state, np_out[_FOCUS])
    np_out[_ASCENSION][: game_state.ascension] = 1.0
    encode_map_meta_into(game_state, np_out[_MAP_META])
    encode_reward_meta_into(game_state, np_out[_REWARD_META])
    encode_shop_meta_into(game_state, np_out[_SHOP_META])
    encode_event_meta_into(game_state, np_out[_EVENT_META])
