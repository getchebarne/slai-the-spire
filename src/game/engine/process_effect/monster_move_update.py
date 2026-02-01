from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_MONSTER


def process_effect_monster_move_update(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]
    ascension_level = kwargs["ascension_level"]

    move_name_new = FACTORY_LIB_MONSTER[target.name](ascension_level)[1](target)
    target.move_name_current = move_name_new
    target.move_name_history.append(move_name_new)

    return [], []
