from src.game.entity.manager import EntityManager
from src.game.entity.manager import add_entity
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.types_ import AscensionLevel


def set_level_exoridium_combat_the_guardian(
    entity_manager: EntityManager, ascension_level: AscensionLevel
) -> None:
    entity_manager.id_monsters = [
        add_entity(entity_manager, FACTORY_LIB_MONSTER["The Guardian"](ascension_level)[0])
    ]

    return
