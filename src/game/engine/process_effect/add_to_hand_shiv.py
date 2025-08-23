from src.game.const import MAX_SIZE_HAND
from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.entity.manager import create_entity
from src.game.factory.lib import FACTORY_LIB_CARD


# TODO: should be public somewhere
_NAME = "Shiv"


def process_effect_add_to_hand_shiv(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    id_cards_in_hand = entity_manager.id_cards_in_hand
    id_cards_in_disc_pile = entity_manager.id_cards_in_disc_pile

    for _ in range(value):
        card = FACTORY_LIB_CARD[_NAME](False)
        id_card = create_entity(entity_manager, card)

        if len(id_cards_in_hand) < MAX_SIZE_HAND:
            # Add to hand
            id_cards_in_hand.append(id_card)
        else:
            # Add to discard pile
            id_cards_in_disc_pile.append(id_card)

    return [], []
