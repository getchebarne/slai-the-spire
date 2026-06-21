import numpy as np
import torch
from slai import Energy


_ENERGY_MIN = 0
_ENERGY_MAX = 5
ENCODING_DIM_ENERGY = _ENERGY_MAX - _ENERGY_MIN + 1 + 1  # Energy OHE  # Energy scalar


def _encode_energy_into(energy: Energy, out: np.ndarray) -> None:
    # Energy OHE
    energy_clamp = min(max(energy.energy_current, _ENERGY_MIN), _ENERGY_MAX)
    out[energy_clamp - _ENERGY_MIN] = 1.0

    # Energy scalar (clamped — energy_current can exceed _ENERGY_MAX via relics)
    out[_ENERGY_MAX - _ENERGY_MIN + 1] = min(energy.energy_current / _ENERGY_MAX, 1.0)


def encode_batch_energy(batch_energy: list[Energy], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_energy)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_ENERGY), dtype=np.float32)

    for b, energy in enumerate(batch_energy):
        _encode_energy_into(energy, np_out[b])

    return torch.from_numpy(np_out).to(device)
