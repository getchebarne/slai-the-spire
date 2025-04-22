import random

from src.game.combat.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_shuffle_deck_into_draw_pile(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_cards_in_draw_pile = entity_manager.id_cards_in_deck.copy()
    random.shuffle(entity_manager.id_cards_in_draw_pile)

    return [], []
