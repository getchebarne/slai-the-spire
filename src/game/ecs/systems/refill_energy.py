from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import RefillEnergyEffect
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class RefillEnergySystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        _ = next(manager.get_components(EffectIsDispatchedComponent, RefillEnergyEffect))

        # TODO: check there's only one energy entity?
        _, energy_component = list(manager.get_component(EnergyComponent))[0]
        energy_component.current = energy_component.max

        return ProcessStatus.COMPLETE
