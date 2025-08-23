from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Footwork"
_COLOR = CardColor.GREEN
_COST = 1
_DEXTERITY_STACKS_GAIN = 2
_DEXTERITY_STACKS_GAIN_PLUS = 3
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.POWER


@register_factory(_NAME)
def create_card_footwork(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(
                    EffectType.MODIFIER_DEXTERITY_GAIN,
                    _DEXTERITY_STACKS_GAIN_PLUS,
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
            Effect(
                EffectType.MODIFIER_DEXTERITY_GAIN,
                _DEXTERITY_STACKS_GAIN,
                EffectTargetType.CHARACTER,
            ),
        ],
    )
