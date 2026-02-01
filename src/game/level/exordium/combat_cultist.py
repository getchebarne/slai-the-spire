from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.types_ import AscensionLevel


def set_level_exoridium_combat_cultist(
    entity_manager: EntityManager, ascension_level: AscensionLevel
) -> None:
    entity_manager.monsters = [FACTORY_LIB_MONSTER["Cultist"](ascension_level)[0]]
