from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.refill_energy import RefillEnergySystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create an energy entity
    max_energy = 3
    current_energy = 1
    energy_entity_id = manager.create_entity(
        EnergyComponent(max=max_energy, current=current_energy)
    )

    # Create effect to refill the energy
    manager.create_entity(EffectRefillEnergy(), EffectIsDispatchedComponent())

    # Run the system
    RefillEnergySystem().process(manager)

    # Assert the current energy is equal to the maximum energy
    energy_component = manager.get_component_for_entity(energy_entity_id, EnergyComponent)
    assert energy_component.current == energy_component.max

    # Assert the maximum energy hasn't changed
    assert energy_component.max == max_energy


def test_current_equal_to_max() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create an energy entity
    max_energy = 3
    current_energy = 3
    energy_entity_id = manager.create_entity(
        EnergyComponent(max=max_energy, current=current_energy)
    )

    # Create effect to refill the energy
    manager.create_entity(EffectRefillEnergy(), EffectIsDispatchedComponent())

    # Run the system
    RefillEnergySystem().process(manager)

    # Assert the current energy is equal to the maximum energy
    energy_component = manager.get_component_for_entity(energy_entity_id, EnergyComponent)
    assert energy_component.current == energy_component.max

    # Assert the maximum energy hasn't changed
    assert energy_component.max == max_energy


def test_current_higher_than_max() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create an energy entity
    max_energy = 3
    current_energy = 4
    energy_entity_id = manager.create_entity(
        EnergyComponent(max=max_energy, current=current_energy)
    )

    # Create effect to refill the energy
    manager.create_entity(EffectRefillEnergy(), EffectIsDispatchedComponent())

    # Run the system
    RefillEnergySystem().process(manager)

    # Assert the current energy is equal to the maximum energy
    energy_component = manager.get_component_for_entity(energy_entity_id, EnergyComponent)
    assert energy_component.current == energy_component.max

    # Assert the maximum energy hasn't changed
    assert energy_component.max == max_energy
