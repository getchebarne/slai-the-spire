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


ENCODING_DIM_ASCENSION = ASCENSION_LEVEL

ENERGY = slice(0, ENCODING_DIM_ENERGY)
FOCUS = slice(ENERGY.stop, ENERGY.stop + ENCODING_DIM_FOCUS)
ASCENSION = slice(FOCUS.stop, FOCUS.stop + ENCODING_DIM_ASCENSION)
MAP_META = slice(ASCENSION.stop, ASCENSION.stop + ENCODING_DIM_MAP_META)
REWARD_META = slice(MAP_META.stop, MAP_META.stop + ENCODING_DIM_REWARD_META)
SHOP_META = slice(REWARD_META.stop, REWARD_META.stop + ENCODING_DIM_SHOP_META)
EVENT_META = slice(SHOP_META.stop, SHOP_META.stop + ENCODING_DIM_EVENT_META)
ENCODING_DIM_CONTEXT = EVENT_META.stop


def focus_view(x_context: torch.Tensor) -> torch.Tensor:
    """View the focus columns without copying or storing a second tensor field."""
    return x_context[:, FOCUS]


def encode_context_into(game_state: GameState, out: np.ndarray) -> None:
    """Write one state's dense features into its initially zeroed context row."""
    encode_energy_into(game_state, out[ENERGY])
    encode_focus_into(game_state, out[FOCUS])
    out[ASCENSION][: game_state.ascension] = 1.0
    encode_map_meta_into(game_state, out[MAP_META])
    encode_reward_meta_into(game_state, out[REWARD_META])
    encode_shop_meta_into(game_state, out[SHOP_META])
    encode_event_meta_into(game_state, out[EVENT_META])
