from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard


COST = 0
DAMAGE = 3
WEAK = 1


def create_card_neutralize() -> EntityCard:
    return EntityCard(
        "Neutralize",
        COST,
        [
            Effect(EffectType.DAMAGE_DEAL, DAMAGE, EffectTargetType.CARD_TARGET),
            Effect(EffectType.MODIFIER_WEAK_GAIN, WEAK, EffectTargetType.CARD_TARGET),
        ],
    )
