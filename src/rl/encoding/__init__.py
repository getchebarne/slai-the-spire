"""Encoding module for converting game state to neural network inputs.

Encoders write observations into reusable CPU buffers. Readers expose their populated prefixes
as the neural network's tokens and per-state fields; allocating batches is a convenience over
the same encoding path.
"""

from src.rl.encoding.buffers import StateBuffer
from src.rl.encoding.buffers import allocate_buffer
from src.rl.encoding.buffers import read_buffer
from src.rl.encoding.state import encode_batch_game_state
from src.rl.encoding.state import encode_into


__all__ = [
    "StateBuffer",
    "allocate_buffer",
    "encode_batch_game_state",
    "encode_into",
    "read_buffer",
]
