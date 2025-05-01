from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_card_discard(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    entity_manager.id_cards_in_hand.remove(id_target)
    entity_manager.id_cards_in_disc_pile.append(id_target)

    return [], []
