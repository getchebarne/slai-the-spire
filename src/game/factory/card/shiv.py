from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Shiv"
_COLOR = CardColor.COLORLESS
_COST = 0
_DAMAGE = 4
_DAMAGE_PLUS = 6
_EXHAUST = True
_RARITY = CardRarity.SPECIAL
_TYPE = CardType.ATTACK


@register_factory(_NAME)
def create_card_shiv(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _DAMAGE_PLUS, EffectTargetType.CARD_TARGET)],
            _EXHAUST,
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _DAMAGE, EffectTargetType.CARD_TARGET)],
        _EXHAUST,
    )
