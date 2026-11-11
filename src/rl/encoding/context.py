"""Dense per-state features in the global projection's established column order."""

import numpy as np
import torch
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


def focus_view(x_context: torch.Tensor) -> torch.Tensor:
    """View the focus columns without copying or storing a second tensor field.

    Parameters
    ----------
    x_context : torch.Tensor
        (B, ENCODING_DIM_CONTEXT). The batch's dense per-state features.

    Returns
    -------
    torch.Tensor
        (B, ENCODING_DIM_FOCUS). A view sharing the context tensor's storage.
    """
    return x_context[:, _FOCUS]


def encode_context_into(game_state: GameState, out: np.ndarray) -> None:
    """Write one state's dense features into its initially zeroed context row."""
    encode_energy_into(game_state, out[_ENERGY])
    encode_focus_into(game_state, out[_FOCUS])
    out[_ASCENSION][: game_state.ascension] = 1.0
    encode_map_meta_into(game_state, out[_MAP_META])
    encode_reward_meta_into(game_state, out[_REWARD_META])
    encode_shop_meta_into(game_state, out[_SHOP_META])
    encode_event_meta_into(game_state, out[_EVENT_META])
