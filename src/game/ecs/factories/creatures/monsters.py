from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import DummyAIComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterPendingMoveUpdateComponent
from src.game.ecs.manager import ECSManager


def create_dummy(manager: ECSManager) -> int:
    base_health = 30

    return manager.create_entity(
        MonsterComponent(0),
        NameComponent("Dummy"),
        DummyAIComponent(),
        HealthComponent(base_health),
        BlockComponent(),
        MonsterPendingMoveUpdateComponent(),
    )
