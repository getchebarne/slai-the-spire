from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.entity.manager import delete_entity


# TODO: upgraded cards
def process_effect_card_reward_select(
    entity_manager: EntityManager, **kwargs
) -> list[tuple[Effect], tuple[Effect]]:
    id_target = kwargs["id_target"]

    # Add card to the deck
    entity_manager.id_cards_in_deck.append(id_target)

    # Clear rewards
    for id_ in entity_manager.id_card_reward:
        if id_ == id_target:
            continue

        delete_entity(entity_manager, id_)

    entity_manager.id_card_reward = []

    return [], []
