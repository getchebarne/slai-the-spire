from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
BLOCK = 5
DRAW = 2


def create_card_backflip() -> EntityCard:
    return EntityCard(
        "Backflip",
        COST,
        [
            Effect(EffectType.BLOCK_GAIN, BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.CARD_DRAW, DRAW),
        ],
    )
