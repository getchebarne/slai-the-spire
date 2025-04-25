from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Backflip"
_COST = 1
_BLOCK = 5
_BLOCK_PLUS = 5
_DRAW = 2


@register_factory(_NAME)
def create_card_backflip(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(EffectType.CARD_DRAW, _DRAW),
            ],
        )

    return EntityCard(
        _NAME,
        _COST,
        [
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.CARD_DRAW, _DRAW),
        ],
    )
