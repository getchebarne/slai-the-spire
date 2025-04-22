from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_block_reset(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]

    target.block_current = 0

    return [], []
