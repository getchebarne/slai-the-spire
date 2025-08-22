import torch

from src.game.view.energy import ViewEnergy
from src.rl.utils import encode_one_hot


_ENERGY_MIN = 0
_ENERGY_MAX = 3


def _get_view_energy_dummy() -> ViewEnergy:
    return ViewEnergy(3, 3)


def get_encoding_energy_dim() -> int:
    view_energy_dummy = _get_view_energy_dummy()
    encoding_energy_dummy = encode_view_energy(view_energy_dummy, torch.device("cpu"))
    return encoding_energy_dummy.shape[0]


# TODO: add max energy, it's always 3 for now
def encode_view_energy(view_energy: ViewEnergy, device: torch.device) -> torch.Tensor:
    return encode_one_hot(view_energy.current, _ENERGY_MIN, _ENERGY_MAX, device)
