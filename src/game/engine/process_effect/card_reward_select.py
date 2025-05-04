from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


# TODO: upgraded cards
def process_effect_card_reward_select(
    entity_manager: EntityManager, **kwargs
) -> list[tuple[Effect], tuple[Effect]]:
    id_target = kwargs["id_target"]

    # Add card to the deck
    entity_manager.id_cards_in_deck.append(id_target)

    # Clear rewards TODO: here?
    entity_manager.id_card_reward = []

    return [], []
