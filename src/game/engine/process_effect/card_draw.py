import random

from src.game.const import MAX_SIZE_HAND
from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


# TODO: handle infinite loop
def process_effect_card_draw(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    draw_pile = entity_manager.draw_pile
    hand = entity_manager.hand
    disc_pile = entity_manager.disc_pile

    for _ in range(value):
        if len(draw_pile) == 0:
            # Shuffle discard pile into draw pile TODO: make effect
            draw_pile.extend(disc_pile)
            random.shuffle(draw_pile)

            # Clear the discard pile
            disc_pile.clear()

        if len(draw_pile) == 0:
            # Both piles are empty, can't draw
            break

        # Remove the top card from the draw pile
        card = draw_pile.pop(0)

        if len(hand) < MAX_SIZE_HAND:
            # Add to hand
            hand.append(card)
        else:
            # Add to discard pile
            disc_pile.append(card)

    return [], []
