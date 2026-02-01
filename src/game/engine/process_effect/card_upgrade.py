from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_CARD


def process_effect_card_upgrade(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    # Create upgraded version of the card
    upgraded_card = FACTORY_LIB_CARD[target.name](upgraded=True)

    # Find and replace in deck
    for i, card in enumerate(entity_manager.deck):
        if card is target:
            entity_manager.deck[i] = upgraded_card
            break

    return [], []
