from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_card_reward_clear(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    """Clear the card reward list (e.g., after skipping a card reward)."""
    entity_manager.card_reward = []

    return [], []
