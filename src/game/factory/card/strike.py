from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
DAMAGE = 6


def create_card_strike() -> EntityCard:
    return EntityCard(
        "Strike",
        COST,
        [Effect(EffectType.DAMAGE_DEAL, DAMAGE, EffectTargetType.CARD_TARGET)],
    )
