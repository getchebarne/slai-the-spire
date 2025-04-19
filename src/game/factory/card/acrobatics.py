from src.game.combat.effect import Effect
from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
DRAW = 3
DISCARD = 1


def create_card_acrobatics() -> EntityCard:
    return EntityCard(
        "Acrobatics",
        COST,
        [
            Effect(EffectType.DRAW_CARD, DRAW),
            Effect(
                EffectType.DISCARD,
                DISCARD,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
