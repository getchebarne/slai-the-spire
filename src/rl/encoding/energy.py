import numpy as np
from slai import GameState

from src.rl.encoding.layout import make_channel_allocator


_ENERGY_MIN = 0
_ENERGY_MAX = 15  # Ice Cream accumulates energy across turns


_CHANNEL_LAYOUT_ENERGY, _alloc = make_channel_allocator()
_OFF_ENERGY = _alloc("energy", _ENERGY_MAX - _ENERGY_MIN + 1)
_OFF_ENERGY_SCALAR = _alloc("energy_scalar")

ENCODING_DIM_ENERGY = sum(w for _, w in _CHANNEL_LAYOUT_ENERGY)


def encode_energy_into(game_state: GameState, out: np.ndarray) -> None:
    """Write the current combat energy, leaving noncombat rows zero."""
    combat = game_state.combat
    if combat is None:
        return
    energy = combat.energy
    energy_clamp = min(max(energy.energy_current, _ENERGY_MIN), _ENERGY_MAX)
    out[_OFF_ENERGY + energy_clamp - _ENERGY_MIN] = 1.0

    # Clamped — energy_current can exceed _ENERGY_MAX via Relics
    out[_OFF_ENERGY_SCALAR] = min(energy.energy_current / _ENERGY_MAX, 1.0)
