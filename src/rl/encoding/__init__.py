"""Encoding module for converting game state to neural network inputs.

Encoders write observations into the reusable CPU arrays they are handed. The package also
exposes the layout names the models and `batches.py` read.
"""

from src.rl.encoding.context import ENCODING_DIM_CONTEXT
from src.rl.encoding.context import focus_view
from src.rl.encoding.effect_layout import ENCODING_DIM_EFFECT
from src.rl.encoding.effect_layout import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.map_ import EDGE_DELTAS
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_EDGE_CHANNELS
from src.rl.encoding.map_ import OFF_MAP_POSITION
from src.rl.encoding.map_ import SLICE_MAP_EDGES
from src.rl.encoding.map_ import SLICE_MAP_ROOM_KIND
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.state import COL_START_CHARACTER
from src.rl.encoding.state import COL_START_HEALTH_BLOCK
from src.rl.encoding.state import COL_START_MONSTER
from src.rl.encoding.state import COL_START_POTION
from src.rl.encoding.state import COL_START_PRICE
from src.rl.encoding.state import COL_START_RELIC
from src.rl.encoding.state import COL_START_STASIS_SLOT
from src.rl.encoding.state import COL_START_TAG
from src.rl.encoding.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.encoding.state import ENCODING_DIM_TOKEN
from src.rl.encoding.state import encode_states_into


__all__ = [
    "encode_states_into",
    "COL_START_CHARACTER",
    "COL_START_HEALTH_BLOCK",
    "COL_START_MONSTER",
    "COL_START_POTION",
    "COL_START_PRICE",
    "COL_START_RELIC",
    "COL_START_STASIS_SLOT",
    "COL_START_TAG",
    "EDGE_DELTAS",
    "ENCODING_DIM_CONTEXT",
    "ENCODING_DIM_EFFECT",
    "ENCODING_DIM_FOCUS",
    "ENCODING_DIM_MODIFIER",
    "ENCODING_DIM_SPARSE_EFFECT",
    "ENCODING_DIM_SPARSE_MODIFIER",
    "ENCODING_DIM_SPARSE_TOKEN",
    "ENCODING_DIM_TOKEN",
    "NUM_CHANNELS",
    "NUM_EDGE_CHANNELS",
    "OFF_FOCUS_DRAW_ORDER_VISIBLE",
    "OFF_MAP_POSITION",
    "SLICE_MAP_EDGES",
    "SLICE_MAP_ROOM_KIND",
    "focus_view",
]
