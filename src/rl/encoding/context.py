"""Dense per-state features in the global projection's established column order."""

import numpy as np
from slai import GameState

from src.rl.encoding.energy import encode_energy_into
from src.rl.encoding.event import encode_event_meta_into
from src.rl.encoding.focus import encode_focus_into
from src.rl.encoding.map_ import encode_map_meta_into
from src.rl.encoding.reward import encode_reward_meta_into
from src.rl.encoding.shop import encode_shop_meta_into
from src.rl.schema.context import ASCENSION
from src.rl.schema.context import ENERGY
from src.rl.schema.context import EVENT_META
from src.rl.schema.context import FOCUS
from src.rl.schema.context import MAP_META
from src.rl.schema.context import REWARD_META
from src.rl.schema.context import SHOP_META


def encode_context_into(game_state: GameState, out: np.ndarray) -> None:
    """Write one state's dense features into its initially zeroed context row."""
    encode_energy_into(game_state, out[ENERGY])
    encode_focus_into(game_state, out[FOCUS])
    out[ASCENSION][: game_state.ascension] = 1.0
    encode_map_meta_into(game_state, out[MAP_META])
    encode_reward_meta_into(game_state, out[REWARD_META])
    encode_shop_meta_into(game_state, out[SHOP_META])
    encode_event_meta_into(game_state, out[EVENT_META])
