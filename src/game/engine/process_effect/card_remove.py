from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.utils import remove_by_identity


def process_effect_card_remove(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    remove_by_identity(entity_manager.hand, target)

    return [], []
