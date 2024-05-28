from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.components.effects import SetBlockToZeroEffect
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.set_block_to_zero import SetBlockToZeroSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a target entity w/ a block component
    max_block = 999
    current_block = 0
    target_entity_id = manager.create_entity(
        BlockComponent(max=max_block, current=current_block), EffectTargetComponent()
    )

    # Create effect to set the entity's block to 0
    manager.create_entity(SetBlockToZeroEffect(), EffectIsDispatchedComponent())

    # Run the system
    SetBlockToZeroSystem().process(manager)

    # Assert the entity's block is 0
    assert manager.get_component_for_entity(target_entity_id, BlockComponent).current == 0

    # Assert the entity's max block hasn't changed
    assert manager.get_component_for_entity(target_entity_id, BlockComponent).max == max_block
