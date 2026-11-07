"""Feature layout and static catalogs for context rows."""

import torch

from src.rl.constants import ASCENSION_LEVEL
from src.rl.schema.energy import ENCODING_DIM_ENERGY
from src.rl.schema.event import ENCODING_DIM_EVENT_META
from src.rl.schema.focus import ENCODING_DIM_FOCUS
from src.rl.schema.map_ import ENCODING_DIM_MAP_META
from src.rl.schema.reward import ENCODING_DIM_REWARD_META
from src.rl.schema.shop import ENCODING_DIM_SHOP_META


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
    return x_context[:, FOCUS]
