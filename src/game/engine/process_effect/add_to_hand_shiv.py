from src.game.const import MAX_SIZE_HAND
from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_CARD


# TODO: should be public somewhere
_NAME = "Shiv"


def process_effect_add_to_hand_shiv(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    for _ in range(value):
        card = FACTORY_LIB_CARD[_NAME](False)

        if len(entity_manager.hand) < MAX_SIZE_HAND:
            # Add to hand
            entity_manager.hand.append(card)
        else:
            # Add to discard pile
            entity_manager.disc_pile.append(card)

    return [], []
