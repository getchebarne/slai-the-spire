from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "A Thousand Cuts"
_COLOR = CardColor.GREEN
_COST = 2
_DAMAGE = 1
_DAMAGE_PLUS = 2
_RARITY = CardRarity.RARE
_TYPE = CardType.POWER


@register_factory(_NAME)
def create_card_a_thousand_cuts(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(
                    EffectType.MODIFIER_THOUSAND_CUTS_GAIN,
                    _DAMAGE_PLUS,
                    EffectTargetType.CHARACTER,
                ),
            ],
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [
            Effect(EffectType.MODIFIER_THOUSAND_CUTS_GAIN, _DAMAGE, EffectTargetType.CHARACTER),
        ],
    )
