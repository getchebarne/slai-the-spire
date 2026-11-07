import numpy as np
from slai import GameState

from src.rl.schema.energy import ENERGY_MAX
from src.rl.schema.energy import ENERGY_MIN
from src.rl.schema.energy import OFF_ENERGY
from src.rl.schema.energy import OFF_ENERGY_SCALAR


def encode_energy_into(game_state: GameState, out: np.ndarray) -> None:
    """Write the current combat energy, leaving noncombat rows zero."""
    combat = game_state.combat
    if combat is None:
        return
    energy = combat.energy
    energy_clamp = min(max(energy.energy_current, ENERGY_MIN), ENERGY_MAX)
    out[OFF_ENERGY + energy_clamp - ENERGY_MIN] = 1.0

    # Clamped — energy_current can exceed ENERGY_MAX via Relics
    out[OFF_ENERGY_SCALAR] = min(energy.energy_current / ENERGY_MAX, 1.0)
