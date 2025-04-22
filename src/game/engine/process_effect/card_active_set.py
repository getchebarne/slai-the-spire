from src.game.combat.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_card_active_set(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_card_active = effect.id_target

    return [], []
