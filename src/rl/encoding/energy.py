import numpy as np
import torch

from src.game.view.energy import ViewEnergy


_ENERGY_MIN = 0
_ENERGY_MAX = 5


def get_encoding_dim_energy() -> int:
    """Calculate energy encoding dimension."""
    return (_ENERGY_MAX - _ENERGY_MIN + 1) + 1  # One-hot + scalar


_ENCODING_DIM_ENERGY = get_encoding_dim_energy()


def _encode_energy_into(out: np.ndarray, energy_current: int) -> None:
    """Encode energy directly into a pre-allocated numpy array."""
    # One-hot
    energy_clamp = max(min(energy_current, _ENERGY_MAX), _ENERGY_MIN)
    out[energy_clamp - _ENERGY_MIN] = 1.0

    # Scalar
    out[_ENERGY_MAX - _ENERGY_MIN + 1] = energy_current / _ENERGY_MAX


def encode_batch_view_energy(
    batch_view_energy: list[ViewEnergy], device: torch.device
) -> torch.Tensor:
    """Encode a batch of energy using NumPy pre-allocation.
    
    Returns: (B, dim_energy) tensor
    """
    batch_size = len(batch_view_energy)

    # Pre-allocate numpy array (2D: batch x features)
    x_out = np.zeros((batch_size, _ENCODING_DIM_ENERGY), dtype=np.float32)

    for b, view_energy in enumerate(batch_view_energy):
        _encode_energy_into(x_out[b], view_energy.current)

    return torch.from_numpy(x_out).to(device)
