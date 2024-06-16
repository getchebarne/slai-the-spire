from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


def create_silent(manager: ECSManager) -> int:
    base_health = 70

    return manager.create_entity(
        ActorComponent(),
        CharacterComponent(),
        NameComponent("Silent"),
        HealthComponent(base_health),
        BlockComponent(),
    )
