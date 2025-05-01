from src.game.entity.manager import EntityManager
from src.game.entity.manager import create_entity
from src.game.factory.lib import FACTORY_LIB_MONSTER


def set_level_exoridium_combat_fungi_beast_two(entity_manager: EntityManager) -> None:
    entity_manager.id_monsters = [
        create_entity(entity_manager, FACTORY_LIB_MONSTER["Fungi Beast"](20)),
        create_entity(entity_manager, FACTORY_LIB_MONSTER["Fungi Beast"](20)),
    ]

    return
