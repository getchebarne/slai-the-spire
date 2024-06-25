from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class RefillEnergySystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            next(manager.get_components(EffectIsTargetedSingletonComponent, EffectRefillEnergy))

        except StopIteration:
            return

        # TODO: check there's only one energy entity?
        _, energy_component = list(manager.get_component(EnergyComponent))[0]
        energy_component.current = energy_component.max

        return
