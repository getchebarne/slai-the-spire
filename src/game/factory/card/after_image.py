from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "After Image"
_COLOR = CardColor.GREEN
_COST = 1
_INNATE = False
_INNATE_PLUS = True
_RARITY = CardRarity.RARE
_TYPE = CardType.POWER


@register_factory(_NAME)
def create_card_after_image(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(EffectType.MODIFIER_AFTER_IMAGE_GAIN, 1, EffectTargetType.CHARACTER),
            ],
            innate=_INNATE_PLUS,
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [
            Effect(EffectType.MODIFIER_AFTER_IMAGE_GAIN, 1, EffectTargetType.CHARACTER),
        ],
        innate=_INNATE,
    )
