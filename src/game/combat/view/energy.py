from dataclasses import dataclass

from src.game.combat.state import Energy
from src.game.combat.state import GameState


@dataclass
class EnergyView:
    current: int
    max: int


def _energy_to_view(energy: Energy) -> EnergyView:
    return EnergyView(energy.current, energy.max)


def view_energy(state: GameState) -> EnergyView:
    return _energy_to_view(state.get_energy())
