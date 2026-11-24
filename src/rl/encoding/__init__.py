"""Encoding module for converting game state to neural network inputs.

Encoders write observations into the reusable CPU arrays they are handed. The package also
exposes the layout names the models and `batches.py` read.
"""

from src.rl.encoding.context import ENCODING_DIM_CONTEXT
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.effect import ENCODING_DIM_EFFECT_SPARSE
from src.rl.encoding.map_ import EDGE_DELTAS
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_FEATURE_CHANNELS
from src.rl.encoding.map_ import OFF_MAP_POSITION
from src.rl.encoding.map_ import OFF_MAP_SELECTABLE
from src.rl.encoding.map_ import SLICE_MAP_EDGES
from src.rl.encoding.map_ import SLICE_MAP_ROOM_KIND
from src.rl.encoding.mask import unpack_legal_bits
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER_SPARSE
from src.rl.encoding.state import COL_START_CHARACTER
from src.rl.encoding.state import COL_START_EVENT_OPTION
from src.rl.encoding.state import COL_START_HEALTH_BLOCK
from src.rl.encoding.state import COL_START_MONSTER
from src.rl.encoding.state import COL_START_POTION
from src.rl.encoding.state import COL_START_PRICE
from src.rl.encoding.state import COL_START_RELIC
from src.rl.encoding.state import COL_START_STASIS_SLOT
from src.rl.encoding.state import COL_START_TAG
from src.rl.encoding.state import ENCODING_DIM_TOKEN
from src.rl.encoding.state import ENCODING_DIM_TOKEN_SPARSE
from src.rl.encoding.state import encode_states_into


__all__ = [
    "encode_states_into",
    "unpack_legal_bits",
    "COL_START_CHARACTER",
    "COL_START_EVENT_OPTION",
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
    "ENCODING_DIM_EFFECT_SPARSE",
    "ENCODING_DIM_MODIFIER",
    "ENCODING_DIM_MODIFIER_SPARSE",
    "ENCODING_DIM_TOKEN",
    "ENCODING_DIM_TOKEN_SPARSE",
    "NUM_CHANNELS",
    "NUM_FEATURE_CHANNELS",
    "OFF_MAP_POSITION",
    "OFF_MAP_SELECTABLE",
    "SLICE_MAP_EDGES",
    "SLICE_MAP_ROOM_KIND",
]
