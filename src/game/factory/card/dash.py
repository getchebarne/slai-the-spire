from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 2
BLOCK = 10
DAMAGE = 10


def create_card_dash() -> EntityCard:
    return EntityCard(
        "Dash",
        COST,
        [
            Effect(EffectType.GAIN_BLOCK, BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.DEAL_DAMAGE, DAMAGE, EffectTargetType.CARD_TARGET),
        ],
    )
