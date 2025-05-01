from src.game.ai.registry import AI_REGISTRY
from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_monster_move_update(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    move_name_new = AI_REGISTRY[target.name](target)
    target.move_name_current = move_name_new
    target.move_name_history.append(move_name_new)

    return [], []
