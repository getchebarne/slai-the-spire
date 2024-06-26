from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.gain_block import GainBlockSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ a block component
    max_block = 999
    current_block = 0
    target_entity_id = manager.create_entity(
        BlockComponent(max=max_block, current=current_block), EffectTargetComponent()
    )

    # Create effect to add `block` block to the entity
    block = 6
    manager.create_entity(EffectGainBlockComponent(block), EffectIsDispatchedComponent())

    # Run the system
    GainBlockSystem().process(manager)

    # Assert the entity's block has increased by `block`
    assert (
        manager.get_component_for_entity(target_entity_id, BlockComponent).current
        == current_block + block
    )

    # Assert the entity's max block hasn't changed
    assert manager.get_component_for_entity(target_entity_id, BlockComponent).max == max_block


def test_capped() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ a block component
    max_block = 999
    current_block = 997
    target_entity_id = manager.create_entity(
        BlockComponent(max=max_block, current=current_block), EffectTargetComponent()
    )

    # Create effect to add `block` block to the entity
    block = 6
    manager.create_entity(EffectGainBlockComponent(block), EffectIsDispatchedComponent())

    # Run the system
    GainBlockSystem().process(manager)

    # Assert the entity's block is capped
    assert manager.get_component_for_entity(target_entity_id, BlockComponent).current == max_block

    # Assert the entity's max block hasn't changed
    assert manager.get_component_for_entity(target_entity_id, BlockComponent).max == max_block
