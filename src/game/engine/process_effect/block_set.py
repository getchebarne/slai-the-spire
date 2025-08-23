from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_block_set(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    target.block_current = value

    return [], []
