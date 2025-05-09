from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Leg Sweep"
_COLOR = CardColor.GREEN
_COST = 2
_BLOCK = 11
_BLOCK_PLUS = 14
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.SKILL
_WEAK = 2
_WEAK_PLUS = 3


@register_factory(_NAME)
def create_card_leg_sweep(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK_PLUS, EffectTargetType.CARD_TARGET),
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
            Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK, EffectTargetType.CARD_TARGET),
        ],
    )
