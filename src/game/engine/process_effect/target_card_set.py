from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_target_card_set(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    entity_manager.id_card_target = id_target

    return [], []
