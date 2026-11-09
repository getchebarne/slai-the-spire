from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Energy
from slai import GameEnv
from slai import GameState
from slai import RelicName

from src.rl.types import make_channel_allocator


_ENERGY_CURRENT_MIN = 0
_ENERGY_CURRENT_MAX = 15  # Ice Cream accumulates energy across turns

# The Relics that add 1 to max energy: the engine's process_effect_combat_start, which GameEnv
# doesn't expose
_RELIC_NAMES_ENERGY = frozenset(
    {
        RelicName.PhilosopherStone,
        RelicName.CoffeeDripper,
        RelicName.FusionHammer,
        RelicName.Sozu,
        RelicName.CursedKey,
        RelicName.BustedCrown,
        RelicName.Ectoplasm,
        RelicName.VelvetChoker,
        RelicName.RunicDome,
    }
)

# Slaver's Collar adds 1 more, in elite and boss fights only
_ENERGY_MAX_CAP = GameEnv.ENERGY_MAX_BASE + len(_RELIC_NAMES_ENERGY) + 1


class EnergyChannel(StrEnum):
    ENERGY = auto()
    ENERGY_SCALAR = auto()
    ENERGY_MAX = auto()
    ENERGY_MAX_SCALAR = auto()  # On ENERGY_SCALAR's scale


CHANNEL_LAYOUT_ENERGY, _alloc = make_channel_allocator()
OFF_ENERGY = _alloc(EnergyChannel.ENERGY, _ENERGY_CURRENT_MAX - _ENERGY_CURRENT_MIN + 1)
OFF_ENERGY_SCALAR = _alloc(EnergyChannel.ENERGY_SCALAR)
OFF_ENERGY_MAX = _alloc(EnergyChannel.ENERGY_MAX, _ENERGY_MAX_CAP - GameEnv.ENERGY_MAX_BASE + 1)
OFF_ENERGY_MAX_SCALAR = _alloc(EnergyChannel.ENERGY_MAX_SCALAR)

ENCODING_DIM_ENERGY = sum(w for _, w in CHANNEL_LAYOUT_ENERGY)


def _encode_energy_into(energy: Energy, out: np.ndarray) -> None:
    energy_clamp = min(max(energy.energy_current, _ENERGY_CURRENT_MIN), _ENERGY_CURRENT_MAX)
    out[OFF_ENERGY + energy_clamp - _ENERGY_CURRENT_MIN] = 1.0

    # Clamped — energy_current can exceed _ENERGY_CURRENT_MAX via Relics
    out[OFF_ENERGY_SCALAR] = min(energy.energy_current / _ENERGY_CURRENT_MAX, 1.0)


def get_energy_max(game_state: GameState) -> int:
    """Return the Character's max energy.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    int
        The Combat's max energy; outside Combat, where the engine holds none, what the owned
        Relics give, short of Slaver's Collar.
    """
    combat = game_state.combat
    if combat is not None:
        return combat.energy.energy_max

    energy_relics = sum(relic.name in _RELIC_NAMES_ENERGY for relic in game_state.relics)
    return GameEnv.ENERGY_MAX_BASE + energy_relics


def _encode_energy_max_into(game_state: GameState, out: np.ndarray) -> None:
    energy_max = get_energy_max(game_state)
    out[OFF_ENERGY_MAX + energy_max - GameEnv.ENERGY_MAX_BASE] = 1.0
    out[OFF_ENERGY_MAX_SCALAR] = energy_max / _ENERGY_CURRENT_MAX


def encode_batch_energy(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    batch_size = len(game_states)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_ENERGY), dtype=np.float32)

    for b, game_state in enumerate(game_states):
        combat = game_state.combat
        if combat is not None:
            _encode_energy_into(combat.energy, np_out[b])

        _encode_energy_max_into(game_state, np_out[b])

    return torch.from_numpy(np_out).to(device)
