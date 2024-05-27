from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.dispatch_effect import DispatchEffectSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create effects
    num_effects = 3
    _ = [manager.create_entity(EffectToBeDispatchedComponent(i)) for i in range(num_effects)]

    # Run the system
    DispatchEffectSystem().process(manager)

    # Verify there's `num_effects` - 1 effects left to be dispatched
    query_result = list(manager.get_component(EffectToBeDispatchedComponent))
    assert len(query_result) == num_effects - 1

    # Assert the priorities of the effects range from 0 to `num_effects` - 2. TODO: improve
    for _, effect_to_be_dispatched_component in query_result:
        assert effect_to_be_dispatched_component.priority in range(0, num_effects - 1)

    # Assert there's one effect that has been dispatched
    assert len(list(manager.get_component(EffectIsDispatchedComponent))) == 1
