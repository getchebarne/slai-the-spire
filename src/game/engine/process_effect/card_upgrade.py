from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_CARD


def process_effect_card_upgrade(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    card = entity_manager.entities[effect.id_target]
    card = FACTORY_LIB_CARD[card.name](upgraded=True)
    entity_manager.entities[effect.id_target] = card

    return [], []
