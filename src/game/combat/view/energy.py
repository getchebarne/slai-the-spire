from dataclasses import dataclass

from src.game.combat.context import Energy
from src.game.combat.context import GameContext


@dataclass
class EnergyView:
    current: int
    max: int


def _energy_to_view(energy: Energy) -> EnergyView:
    return EnergyView(energy.current, energy.max)


def view_energy(context: GameContext) -> EnergyView:
    return _energy_to_view(context.energy)
