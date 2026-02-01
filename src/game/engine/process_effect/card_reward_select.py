from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


# TODO: upgraded cards
def process_effect_card_reward_select(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    # Add card to the deck
    entity_manager.deck.append(target)

    # Clear rewards TODO: here?
    entity_manager.card_reward = []

    return [], []
