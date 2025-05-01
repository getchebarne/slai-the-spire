from src.game.entity.manager import EntityManager
from src.game.entity.manager import create_entity
from src.game.factory.lib import FACTORY_LIB_MONSTER


def set_level_exoridium_combat_jaw_worm(entity_manager: EntityManager) -> None:
    entity_manager.id_monsters = [
        create_entity(entity_manager, FACTORY_LIB_MONSTER["Jaw Worm"](20))
    ]

    return
