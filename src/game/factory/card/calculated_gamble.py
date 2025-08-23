from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Calculated Gamble"
_COLOR = CardColor.GREEN
_COST = 0
_EXHAUST = True
_EXHAUST_PLUS = False
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_calculated_gamble(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [Effect(EffectType.CALCULATED_GAMBLE)],
            _EXHAUST_PLUS,
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [Effect(EffectType.CALCULATED_GAMBLE)],
        _EXHAUST,
    )
