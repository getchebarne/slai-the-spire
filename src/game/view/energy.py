from dataclasses import dataclass

from src.game.entity.manager import EntityManager


@dataclass(frozen=True)
class ViewEnergy:
    current: int
    max: int


def get_view_energy(entity_manager: EntityManager) -> ViewEnergy:
    energy = entity_manager.entities[entity_manager.id_energy]

    return ViewEnergy(energy.current, energy.max)
