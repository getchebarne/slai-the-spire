from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.destroy_effect import DestroyEffectSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create dispatched effect
    effect_entity_id = manager.create_entity(EffectIsDispatchedComponent())

    # Run the system
    DestroyEffectSystem().process(manager)

    # Verify the effect no longer exists
    assert effect_entity_id not in manager._entities
