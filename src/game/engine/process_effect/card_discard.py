from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.utils import remove_by_identity


def process_effect_card_discard(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    remove_by_identity(entity_manager.hand, target)
    entity_manager.disc_pile.append(target)

    return [], []
