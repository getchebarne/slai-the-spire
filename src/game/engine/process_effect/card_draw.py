import random

from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


# TODO: handle infinite loop
# TODO: add max cards
def process_effect_card_draw(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    id_cards_in_draw_pile = entity_manager.id_cards_in_draw_pile
    id_cards_in_hand = entity_manager.id_cards_in_hand
    id_cards_in_disc_pile = entity_manager.id_cards_in_disc_pile

    for _ in range(value):
        if len(id_cards_in_draw_pile) == 0:
            # Shuffle discard pile into draw pile TODO: make effect
            id_cards_in_draw_pile.extend(id_cards_in_disc_pile)
            random.shuffle(id_cards_in_draw_pile)

            # Clear the discard pile
            id_cards_in_disc_pile.clear()

        # Draw a card from the draw pile and add to hand
        id_cards_in_hand.append(id_cards_in_draw_pile.pop(0))

    return [], []
