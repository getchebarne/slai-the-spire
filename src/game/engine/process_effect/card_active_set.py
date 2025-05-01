from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_card_active_set(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    entity_manager.id_card_active = id_target

    return [], []
