from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_block_reset(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    target.block_current = 0

    return [], []
