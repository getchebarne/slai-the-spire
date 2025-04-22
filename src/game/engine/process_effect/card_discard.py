from src.game.combat.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_card_discard(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_cards_in_hand.remove(effect.id_target)
    entity_manager.id_cards_in_disc_pile.append(effect.id_target)

    return [], []
