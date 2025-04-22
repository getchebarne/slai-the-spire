from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_target_effect_clear(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_effect_target = None

    return [], []
