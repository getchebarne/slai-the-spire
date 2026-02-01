import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_combat_start(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:

    # Get innate vs. not innate cards
    cards_innate = []
    cards_not_innate = []
    for card in entity_manager.deck:
        if card.innate:
            cards_innate.append(card)
        else:
            cards_not_innate.append(card)

    # Put innate cards at the beginning of the draw pile
    entity_manager.draw_pile = cards_innate.copy()

    # Shuffle rest of cards and append them to the draw pile
    random.shuffle(cards_not_innate)
    entity_manager.draw_pile += cards_not_innate

    return [], [
        *[Effect(EffectType.MONSTER_MOVE_UPDATE, target=monster) for monster in entity_manager.monsters],
        Effect(EffectType.TURN_START, target=entity_manager.character),
    ]
