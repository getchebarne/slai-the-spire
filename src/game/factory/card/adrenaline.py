from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Adrenaline"
_COLOR = CardColor.GREEN
_COST = 0
_DRAW = 2
_ENERGY_GAIN = 1
_ENERGY_GAIN_PLUS = 2
_EXHAUST = True
_RARITY = CardRarity.RARE
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_adrenaline(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [Effect(EffectType.ENERGY_GAIN, _ENERGY_GAIN), Effect(EffectType.CARD_DRAW, _DRAW)],
            _EXHAUST,
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [Effect(EffectType.ENERGY_GAIN, _ENERGY_GAIN_PLUS), Effect(EffectType.CARD_DRAW, _DRAW)],
        _EXHAUST,
    )
