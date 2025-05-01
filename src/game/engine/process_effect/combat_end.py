from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_combat_end(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    # Clear hand, draw pile and discard pile
    entity_manager.id_cards_in_hand = []
    entity_manager.id_cards_in_draw_pile = []
    entity_manager.id_cards_in_disc_pile = []

    # TODO: will need to add end of combat triggers, such as "Blood Vial"
    return [], []
