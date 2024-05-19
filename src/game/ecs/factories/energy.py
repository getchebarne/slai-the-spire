from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager


def create_energy(manager: ECSManager) -> int:
    base_energy = 3

    return manager.create_entity(EnergyComponent(base_energy))
