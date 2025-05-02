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


_NAME = "Survivor"
_COLOR = CardColor.GREEN
_COST = 1
_BLOCK = 8
_BLOCK_PLUS = 11
_CARD_DISCARD = 1
_RARITY = CardRarity.BASIC
_TYPE = CardType.SKILL


@register_factory(_NAME)
def create_card_survivor(upgraded: CardUpgraded) -> EntityCard:
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
                    EffectType.CARD_DISCARD,
                    _CARD_DISCARD,  # TODO: this should be part of the selection type
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
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(
                EffectType.CARD_DISCARD,
                _CARD_DISCARD,  # TODO: this should be part of the selection type
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
