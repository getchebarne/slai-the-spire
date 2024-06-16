from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import CreatureComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.manager import ECSManager


def create_silent(manager: ECSManager) -> int:
    base_health = 70

    return manager.create_entity(
        CreatureComponent(),
        CharacterComponent(),
        NameComponent("Silent"),
        HealthComponent(base_health),
        BlockComponent(),
    )
