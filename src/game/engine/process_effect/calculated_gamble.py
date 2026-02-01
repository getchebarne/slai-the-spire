from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_calculated_gamble(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    num_cards_in_hand = len(entity_manager.hand)

    return [], [
        Effect(EffectType.CARD_DISCARD, target_type=EffectTargetType.CARD_IN_HAND),
        Effect(EffectType.CARD_DRAW, num_cards_in_hand),
    ]
