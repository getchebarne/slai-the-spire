"""Feature layouts and dimensions shared by encoders, models and worker storage."""

from src.rl.schema.context import ENCODING_DIM_CONTEXT
from src.rl.schema.effect import ENCODING_DIM_EFFECT
from src.rl.schema.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.schema.modifier import ENCODING_DIM_MODIFIER
from src.rl.schema.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.schema.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.schema.state import ENCODING_DIM_TOKEN


__all__ = [
    "ENCODING_DIM_CONTEXT",
    "ENCODING_DIM_EFFECT",
    "ENCODING_DIM_SPARSE_EFFECT",
    "ENCODING_DIM_MODIFIER",
    "ENCODING_DIM_SPARSE_MODIFIER",
    "ENCODING_DIM_TOKEN",
    "ENCODING_DIM_SPARSE_TOKEN",
]
