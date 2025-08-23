import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_combat_start(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:

    # Get innate vs. not innate cards
    id_cards_innate = []
    id_cards_not_innate = []
    for id_card in entity_manager.id_cards_in_deck:
        card = entity_manager.entities[id_card]
        if card.innate:
            id_cards_innate.append(id_card)
        else:
            id_cards_not_innate.append(id_card)

    # Put innate cards at the beginning of the draw pile
    entity_manager.id_cards_in_draw_pile = id_cards_innate

    # Shuffle rest of cards and append them to the draw pile
    random.shuffle(id_cards_not_innate)
    entity_manager.id_cards_in_draw_pile += id_cards_not_innate

    return [], [
        *[
            Effect(EffectType.MONSTER_MOVE_UPDATE, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ],
    ] + [Effect(EffectType.TURN_START, id_target=entity_manager.id_character)]
