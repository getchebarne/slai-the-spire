from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Energy

from src.rl.types import make_channel_allocator


_ENERGY_MIN = 0
_ENERGY_MAX = 15  # Ice Cream accumulates energy accross turns


class EnergyChannel(StrEnum):
    ENERGY = auto()
    ENERGY_SCALAR = auto()


CHANNEL_LAYOUT_ENERGY, _alloc = make_channel_allocator()
OFF_ENERGY = _alloc(EnergyChannel.ENERGY, _ENERGY_MAX - _ENERGY_MIN + 1)
OFF_ENERGY_SCALAR = _alloc(EnergyChannel.ENERGY_SCALAR)

ENCODING_DIM_ENERGY = sum(w for _, w in CHANNEL_LAYOUT_ENERGY)


def _encode_energy_into(energy: Energy, out: np.ndarray) -> None:
    energy_clamp = min(max(energy.energy_current, _ENERGY_MIN), _ENERGY_MAX)
    out[OFF_ENERGY + energy_clamp - _ENERGY_MIN] = 1.0

    # Clamped — energy_current can exceed _ENERGY_MAX via relics
    out[OFF_ENERGY_SCALAR] = min(energy.energy_current / _ENERGY_MAX, 1.0)


def encode_batch_energy(batch_energy: list[Energy | None], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_energy)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_ENERGY), dtype=np.float32)

    for b, energy in enumerate(batch_energy):
        if energy is not None:
            _encode_energy_into(energy, np_out[b])

    return torch.from_numpy(np_out).to(device)
