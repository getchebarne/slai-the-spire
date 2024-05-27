from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.deal_damage import DealDamageSystem


def test_wo_block() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ health and block components
    max_health = 50
    target_entity_id = manager.create_entity(HealthComponent(max_health), BlockComponent())

    # Create effect to deal `damage` damage to the entity
    damage = 6
    manager.create_entity(
        DealDamageEffectComponent(damage),
        EffectQueryComponentsComponent([HealthComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),
        EffectIsDispatchedComponent(),
    )

    # Run the system
    DealDamageSystem().process(manager)

    # Assert the entity's health has decreased by `damage`
    assert (
        manager.get_component_for_entity(target_entity_id, HealthComponent).current
        == max_health - damage
    )

    # Assert the entity's max health hasn't changed
    assert manager.get_component_for_entity(target_entity_id, HealthComponent).max == max_health


def test_w_block_lower_than_damage() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ health and block components
    max_health = 50
    current_block = 4
    target_entity_id = manager.create_entity(
        HealthComponent(max_health), BlockComponent(current=current_block)
    )

    # Create effect to deal `damage` damage to the entity
    damage = 6
    manager.create_entity(
        DealDamageEffectComponent(damage),
        EffectQueryComponentsComponent([HealthComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),
        EffectIsDispatchedComponent(),
    )

    # Run the system
    DealDamageSystem().process(manager)

    # Assert the entity's health has decreased by `damage` - `current_block`
    assert (
        manager.get_component_for_entity(target_entity_id, HealthComponent).current
        == max_health + current_block - damage
    )

    # Assert the entity's max health hasn't changed
    assert manager.get_component_for_entity(target_entity_id, HealthComponent).max == max_health


def test_w_block_higher_than_damage() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ health and block components
    max_health = 50
    current_block = 8
    target_entity_id = manager.create_entity(
        HealthComponent(max_health), BlockComponent(current=current_block)
    )

    # Create effect to deal `damage` damage to the entity
    damage = 6
    manager.create_entity(
        DealDamageEffectComponent(damage),
        EffectQueryComponentsComponent([HealthComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),
        EffectIsDispatchedComponent(),
    )

    # Run the system
    DealDamageSystem().process(manager)

    # Assert the entity's health hasn't decreased
    assert (
        manager.get_component_for_entity(target_entity_id, HealthComponent).current == max_health
    )

    # Assert the entity's max health hasn't changed
    assert manager.get_component_for_entity(target_entity_id, HealthComponent).max == max_health

    # Assert the entity's block has decreased by `damage`
    assert (
        manager.get_component_for_entity(target_entity_id, BlockComponent).current
        == current_block - damage
    )
