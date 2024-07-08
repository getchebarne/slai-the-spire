from dataclasses import dataclass

from src.game.combat.entities import Energy
from src.game.combat.entities import Entities


@dataclass
class EnergyView:
    current: int
    max: int


def _energy_to_view(energy: Energy) -> EnergyView:
    return EnergyView(energy.current, energy.max)


def view_energy(entities: Entities) -> EnergyView:
    return _energy_to_view(entities.get_energy())
