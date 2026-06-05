"""
Encoding module for converting game state to neural network inputs.

This module provides functions to encode various game entities (cards, monsters,
character, energy, map) into tensor representations suitable for the neural network.
"""

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.character import _ENCODING_DIM_CHARACTER
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.energy import _ENCODING_DIM_ENERGY
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import _ENCODING_DIM_EVENT_META
from src.rl.encoding.event import _ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.event import encode_batch_events
from src.rl.encoding.map_ import _NUM_CHANNELS
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.map_ import encode_batch_map_meta
from src.rl.encoding.monster import _ENCODING_DIM_MONSTER
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.reward import _ENCODING_DIM_REWARD_META
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.screen import _ENCODING_DIM_SCREEN
from src.rl.encoding.screen import encode_batch_screen
from src.rl.encoding.shop import _DIM_SHOP_CARD
from src.rl.encoding.shop import _DIM_SHOP_META
from src.rl.encoding.shop import _DIM_SHOP_POTION
from src.rl.encoding.shop import _DIM_SHOP_RELIC
from src.rl.encoding.shop import encode_batch_shop
from src.rl.encoding.state import TensorGameState
from src.rl.encoding.state import encode_batch_game_state


__all__ = [
    # Card encoding
    "encode_batch_cards",
    "ENCODING_DIM_CARD",
    # Character encoding
    "encode_batch_character",
    "_ENCODING_DIM_CHARACTER",
    # Energy encoding
    "encode_batch_energy",
    "_ENCODING_DIM_ENERGY",
    # Screen encoding
    "encode_batch_screen",
    "_ENCODING_DIM_SCREEN",
    # Map encoding
    "encode_batch_map",
    "_NUM_CHANNELS",
    "encode_batch_map_meta",
    "ENCODING_DIM_MAP_META",
    # Monster encoding
    "encode_batch_monsters",
    "_ENCODING_DIM_MONSTER",
    # Potion belt encoding
    "encode_batch_potions",
    "ENCODING_DIM_POTION",
    # Relic encoding
    "encode_batch_relics",
    "ENCODING_DIM_RELIC",
    # Reward-meta encoding
    "encode_batch_rewards",
    "_ENCODING_DIM_REWARD_META",
    # Shop encoding
    "encode_batch_shop",
    "_DIM_SHOP_CARD",
    "_DIM_SHOP_RELIC",
    "_DIM_SHOP_POTION",
    "_DIM_SHOP_META",
    # Event encoding
    "encode_batch_events",
    "_ENCODING_DIM_EVENT_META",
    "_ENCODING_DIM_EVENT_OPTION",
    # Full state encoding
    "encode_batch_game_state",
    "TensorGameState",
]
