from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Infinite Blades"
_COLOR = CardColor.GREEN
_COST = 1
_INFINITE_BLADES_STACKS_CURRENT = 1
_INNATE = False
_INNATE_PLUS = True
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.POWER


@register_factory(_NAME)
def create_card_infinite_blades(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(
                    EffectType.MODIFIER_INFINITE_BLADES_GAIN,
                    _INFINITE_BLADES_STACKS_CURRENT,
                    EffectTargetType.CHARACTER,
                ),
            ],
            innate=_INNATE_PLUS,
        )

    return EntityCard(
        _NAME,
        _COLOR,
        _TYPE,
        _COST,
        _RARITY,
        [
            Effect(
                EffectType.MODIFIER_INFINITE_BLADES_GAIN,
                _INFINITE_BLADES_STACKS_CURRENT,
                EffectTargetType.CHARACTER,
            ),
        ],
        innate=_INNATE,
    )
