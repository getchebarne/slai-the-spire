from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Blade Dance"
_COLOR = CardColor.GREEN
_COST = 1
_NUMBER_OF_SHIVS = 3
_NUMBER_OF_SHIVS_PLUS = 4
_RARITY = CardRarity.COMMON
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_blade_dance(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [Effect(EffectType.ADD_TO_HAND_SHIV, _NUMBER_OF_SHIVS_PLUS)],
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [Effect(EffectType.ADD_TO_HAND_SHIV, _NUMBER_OF_SHIVS)],
    )
