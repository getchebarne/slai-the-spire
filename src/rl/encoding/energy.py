import numpy as np
from slai import GameEnv
from slai import GameState
from slai import Relic
from slai import RelicName

from src.rl.encoding.layout import make_channel_allocator
from src.rl.tokens import get_entities


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


_CHANNEL_LAYOUT_ENERGY, _alloc = make_channel_allocator()
_OFF_ENERGY = _alloc("energy", _ENERGY_CURRENT_MAX - _ENERGY_CURRENT_MIN + 1)
_OFF_ENERGY_SCALAR = _alloc("energy_scalar")
_OFF_ENERGY_MAX = _alloc("energy_max", _ENERGY_MAX_CAP - GameEnv.ENERGY_MAX_BASE + 1)
_OFF_ENERGY_MAX_SCALAR = _alloc("energy_max_scalar")

ENCODING_DIM_ENERGY = sum(w for _, w in _CHANNEL_LAYOUT_ENERGY)


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

    relics = get_entities(game_state, game_state.id_relics, Relic)
    energy_relics = sum(relic.name in _RELIC_NAMES_ENERGY for relic in relics)

    return GameEnv.ENERGY_MAX_BASE + energy_relics


def encode_energy_into(game_state: GameState, np_out: np.ndarray) -> None:
    """Write the max energy, and in Combat the current energy.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    np_out : np.ndarray
        (ENCODING_DIM_ENERGY,) float32. The context row's energy range, written in place; the
        current energy's channels stay 0 outside Combat.
    """
    energy_max = get_energy_max(game_state)
    np_out[_OFF_ENERGY_MAX + energy_max - GameEnv.ENERGY_MAX_BASE] = 1.0
    np_out[_OFF_ENERGY_MAX_SCALAR] = energy_max / _ENERGY_CURRENT_MAX

    combat = game_state.combat
    if combat is None:
        return
    energy = combat.energy
    energy_clamp = min(max(energy.energy_current, _ENERGY_CURRENT_MIN), _ENERGY_CURRENT_MAX)
    np_out[_OFF_ENERGY + energy_clamp - _ENERGY_CURRENT_MIN] = 1.0

    # Clamped — energy_current can exceed _ENERGY_CURRENT_MAX via Relics
    np_out[_OFF_ENERGY_SCALAR] = min(energy.energy_current / _ENERGY_CURRENT_MAX, 1.0)
