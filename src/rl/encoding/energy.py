import torch

from src.game.view.energy import ViewEnergy
from src.rl.utils import encode_one_hot_list


_ENERGY_MIN = 0
_ENERGY_MAX = 5


def get_encoding_dim_energy() -> int:
    view_energy_dummy = ViewEnergy(3, 3)
    encoding_energy_dummy = _encode_energy_current(view_energy_dummy.current)
    return len(encoding_energy_dummy)


def _encode_energy_current(energy_current: int) -> list[int]:
    energy_current_ohe = encode_one_hot_list(energy_current, _ENERGY_MIN, _ENERGY_MAX)

    # Append scalar
    return energy_current_ohe + [energy_current / _ENERGY_MAX]


def encode_batch_view_energy(
    batch_view_energy: list[ViewEnergy], device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [_encode_energy_current(view_energy.current) for view_energy in batch_view_energy],
        dtype=torch.float32,
        device=device,
    )
