from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Dagger Throw"
_COLOR = CardColor.GREEN
_COST = 1
_DAMAGE = 9
_DAMAGE_PLUS = 12
_DRAW = 1
_CARD_DISCARD = 1
_RARITY = CardRarity.COMMON
_TYPE = CardType.ATTACK


@register_factory(_NAME)
def create_card_dagger_throw(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(
                    EffectType.DAMAGE_DEAL_PHYSICAL, _DAMAGE_PLUS, EffectTargetType.CARD_TARGET
                ),
                Effect(EffectType.CARD_DRAW, _DRAW),
                Effect(
                    EffectType.CARD_DISCARD,
                    _CARD_DISCARD,
                    EffectTargetType.CARD_IN_HAND,
                    EffectSelectionType.INPUT,
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
            Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _DAMAGE, EffectTargetType.CARD_TARGET),
            Effect(EffectType.CARD_DRAW, _DRAW),
            Effect(
                EffectType.CARD_DISCARD,
                _CARD_DISCARD,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
