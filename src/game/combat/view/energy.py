from dataclasses import dataclass

from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager


@dataclass
class EnergyView:
    current: int
    max: int


def get_energy_view(manager: ECSManager) -> EnergyView:
    _, energy_component = list(manager.get_component(EnergyComponent))[0]

    return EnergyView(energy_component.current, energy_component.max)
