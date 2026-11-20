"""Identify distinct Maps by their feature grids."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.rl.batches import get_unique_rep_row
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding import NUM_FEATURE_CHANNELS
from src.rl.types import Live
from src.rl.types import States


_NUM_CELLS = MAP_HEIGHT * MAP_WIDTH
_GRID_BITS = _NUM_CELLS * NUM_FEATURE_CHANNELS
_WORD_BITS = 63                                   # An int64 without its sign bit
_NUM_WORDS = math.ceil(_GRID_BITS / _WORD_BITS)
_PAD_BITS = _NUM_WORDS * _WORD_BITS - _GRID_BITS  # Zero bits that fill the last word


@dataclass
class UniqueMaps:
    """The distinct feature grids in a batch, in sorted order."""

    x_state_map: torch.Tensor  # (B,) long. Each state's Map ID.
    x_cell: torch.Tensor       # (U, H * W, NUM_FEATURE_CHANNELS) uint8. Each distinct Map.


def get_unique_maps(states: States[Live]) -> UniqueMaps:
    """Group equal feature grids and return the first grid for each Map ID."""
    x_cell = states.x_map_grid.flatten(1, 2)
    x_unique, x_state_map = torch.unique(_pack_rows(x_cell), dim=0, return_inverse=True)
    x_rep = get_unique_rep_row(x_state_map, x_unique.shape[0])

    return UniqueMaps(x_state_map=x_state_map, x_cell=x_cell[x_rep])


def _pack_rows(x_cell: torch.Tensor) -> torch.Tensor:
    """Pack each grid into int64 words that sort in the same order as the grid's bits."""
    batch_size = x_cell.shape[0]
    x_bit_shift = torch.arange(_WORD_BITS - 1, -1, -1, device=x_cell.device)
    x_bit = F.pad(x_cell.reshape(batch_size, _GRID_BITS).long(), (0, _PAD_BITS))

    return (x_bit.view(batch_size, _NUM_WORDS, _WORD_BITS) << x_bit_shift).sum(-1)
