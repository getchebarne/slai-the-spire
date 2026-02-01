from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_target_card_clear(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.card_target = None

    return [], []
