from src.game.combat.effect import Effect
from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
DAMAGE = 9
DRAW = 1
DISCARD = 1


def create_card_dagger_throw() -> EntityCard:
    return EntityCard(
        "Dagger Throw",
        COST,
        [
            Effect(EffectType.DEAL_DAMAGE, DAMAGE, EffectTargetType.CARD_TARGET),
            Effect(EffectType.DRAW_CARD, DRAW),
            Effect(
                EffectType.DISCARD,
                DISCARD,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
