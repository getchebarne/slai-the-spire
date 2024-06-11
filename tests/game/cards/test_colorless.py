from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardIsSelectedComponent
from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import HasEffectsComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.components.target import TargetComponent
from src.game.ecs.factories.cards.colorless import create_defend
from src.game.ecs.factories.cards.colorless import create_strike
from src.game.ecs.factories.creatures.characters import create_silent
from src.game.ecs.factories.creatures.monsters import create_dummy
from src.game.ecs.factories.energy import create_energy
from src.game.ecs.factories.systems import create_systems
from src.game.ecs.manager import ECSManager


SYSTEMS = create_systems()


def test_strike() -> None:
    # Instantiate an ECS manager
    manager = ECSManager()

    # Create Dummy & Energy entities
    dummy_entity_id = create_dummy(manager)
    energy_entity_id = create_energy(manager)

    # Create a Strike card entity & set it as active
    strike_entity_id = create_strike(manager)
    manager.add_component(strike_entity_id, CardIsSelectedComponent())

    # Get the Strike card's effect entity ids
    effect_entity_ids = manager.get_component_for_entity(
        strike_entity_id, HasEffectsComponent
    ).entity_ids

    # Assert the Strike card has a single effect
    assert len(effect_entity_ids) == 1

    # Select the Dummmy as the target
    manager.add_component(dummy_entity_id, TargetComponent())

    # Store the Dummy's health & Energy before running the systems
    prev_dummy_health = manager.get_component_for_entity(dummy_entity_id, HealthComponent).current
    prev_energy = manager.get_component_for_entity(energy_entity_id, EnergyComponent).current

    # Run the systems
    for system in SYSTEMS:
        system(manager)

    # Assert the Dummy's health has been reduced by Strike's base damage
    base_damage = manager.get_component_for_entity(
        effect_entity_ids[0], EffectDealDamageComponent
    ).value
    assert (
        manager.get_component_for_entity(dummy_entity_id, HealthComponent).current
        == prev_dummy_health - base_damage
    )
    # Assert the Energy has been reduced by Strike's base cost
    base_cost = manager.get_component_for_entity(strike_entity_id, CardCostComponent).value
    assert (
        manager.get_component_for_entity(energy_entity_id, EnergyComponent).current
        == prev_energy - base_cost
    )


def test_defend() -> None:
    # Instantiate an ECS manager
    manager = ECSManager()

    # Create Character & Energy entities
    char_entity_id = create_silent(manager)
    energy_entity_id = create_energy(manager)

    # Create a Defend card entity & set it as active
    defend_entity_id = create_defend(manager)
    manager.add_component(defend_entity_id, CardIsSelectedComponent())

    # Get the Defend card's effect entity ids
    effect_entity_ids = manager.get_component_for_entity(
        defend_entity_id, HasEffectsComponent
    ).entity_ids

    # Assert the Defend card has a single effect
    assert len(effect_entity_ids) == 1

    # Store the Character's block & Energy before running the systems
    prev_char_block = manager.get_component_for_entity(char_entity_id, BlockComponent).current
    prev_energy = manager.get_component_for_entity(energy_entity_id, EnergyComponent).current

    # Run the systems
    for system in SYSTEMS:
        system(manager)

    # Assert the Character's block has been increased by Defend's base block
    base_block = manager.get_component_for_entity(
        effect_entity_ids[0], EffectGainBlockComponent
    ).value
    assert (
        manager.get_component_for_entity(char_entity_id, BlockComponent).current
        == prev_char_block + base_block
    )
    # Assert the Energy has been reduced by Defend's base cost
    base_cost = manager.get_component_for_entity(defend_entity_id, CardCostComponent).value
    assert (
        manager.get_component_for_entity(energy_entity_id, EnergyComponent).current
        == prev_energy - base_cost
    )
