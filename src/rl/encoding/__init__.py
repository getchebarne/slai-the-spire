"""
Encoding module for converting game state to neural network inputs.

This module provides functions to encode various game entities (cards, monsters,
character, energy, map) into tensor representations suitable for the neural network.
"""

from src.rl.encoding.card import CardPile
from src.rl.encoding.card import encode_batch_view_cards
from src.rl.encoding.card import get_encoding_dim_card
from src.rl.encoding.character import encode_batch_view_character
from src.rl.encoding.character import get_encoding_dim_character
from src.rl.encoding.energy import encode_batch_view_energy
from src.rl.encoding.energy import get_encoding_dim_energy
from src.rl.encoding.map_ import encode_batch_view_map
from src.rl.encoding.map_ import get_encoding_map_dim
from src.rl.encoding.monster import encode_batch_view_monsters
from src.rl.encoding.monster import get_encoding_dim_monster
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state


__all__ = [
    # Card encoding
    "CardPile",
    "encode_batch_view_cards",
    "get_encoding_dim_card",
    # Character encoding
    "encode_batch_view_character",
    "get_encoding_dim_character",
    # Energy encoding
    "encode_batch_view_energy",
    "get_encoding_dim_energy",
    # Map encoding
    "encode_batch_view_map",
    "get_encoding_map_dim",
    # Monster encoding
    "encode_batch_view_monsters",
    "get_encoding_dim_monster",
    # Full state encoding
    "encode_batch_view_game_state",
    "XGameState",
]
