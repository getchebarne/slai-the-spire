from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Dodge And Roll"
_COLOR = CardColor.GREEN
_COST = 1
_BLOCK = 4
_BLOCK_PLUS = 6
_RARITY = CardRarity.COMMON
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_dodge_and_roll(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(
                    EffectType.MODIFIER_NEXT_TURN_BLOCK_GAIN,
                    _BLOCK_PLUS,
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
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.MODIFIER_NEXT_TURN_BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
        ],
    )
