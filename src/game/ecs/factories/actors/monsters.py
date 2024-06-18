from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.actors import DummyAIComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


def create_dummy(manager: ECSManager) -> int:
    base_health = 30

    return manager.create_entity(
        NameComponent("Dummy"),
        ActorComponent(),
        MonsterComponent(0),
        DummyAIComponent(),
        HealthComponent(base_health),
        BlockComponent(),
        MonsterPendingMoveUpdateComponent(),
    )
    # manager.create_entity(
    #     MonsterMoveComponent(),
    #     MonsterMoveDummyAttackComponent(),
    #     MonsterMoveIntentComponent(damage=base_damage, times=1, block=False),
    #     MonsterMoveParentComponent(dummy_entity_id),
    # )
    # manager.create_entity(
    #     MonsterMoveComponent(),
    #     MonsterMoveDummyDefendComponent(),
    #     MonsterMoveIntentComponent(damage=0, times=0, block=True),
    #     MonsterMoveParentComponent(dummy_entity_id),
    # )
    # return dummy_entity_id
