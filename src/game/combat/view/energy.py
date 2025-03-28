from dataclasses import dataclass

from src.game.combat.entities import Energy
from src.game.combat.entities import EntityManager


@dataclass
class EnergyView:
    current: int
    max: int


def _energy_to_view(energy: Energy) -> EnergyView:
    return EnergyView(energy.current, energy.max)


def view_energy(entity_manager: EntityManager) -> EnergyView:
    return _energy_to_view(entity_manager.entities[entity_manager.id_energy])
