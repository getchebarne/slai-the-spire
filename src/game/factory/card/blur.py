from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Blur"
_COLOR = CardColor.GREEN
_COST = 1
_BLOCK = 5
_BLOCK_PLUS = 8
_BLUR_GAIN = 1
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_blur(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(EffectType.MODIFIER_BLUR_GAIN, _BLUR_GAIN, EffectTargetType.CHARACTER),
            ],
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.MODIFIER_BLUR_GAIN, _BLUR_GAIN, EffectTargetType.CHARACTER),
        ],
    )
