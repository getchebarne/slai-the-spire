"""Encoding module for converting game state to neural network inputs.

Encoders write observations into the reusable CPU arrays they are handed.
"""

from src.rl.encoding.state import encode_states_into


__all__ = [
    "encode_states_into",
]
