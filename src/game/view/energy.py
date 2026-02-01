from dataclasses import dataclass

from src.game.entity.manager import EntityManager


@dataclass(frozen=True)
class ViewEnergy:
    current: int
    max: int


def get_view_energy(entity_manager: EntityManager) -> ViewEnergy:
    return ViewEnergy(entity_manager.energy.current, entity_manager.energy.max)
