from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Acrobatics"
_COST = 1
_DRAW = 3
_DRAW_PLUS = 3
_CARD_DISCARD = 1


@register_factory(_NAME)
def create_card_acrobatics(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [
                Effect(EffectType.CARD_DRAW, _DRAW_PLUS),
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
        _COST,
        [
            Effect(EffectType.CARD_DRAW, _DRAW),
            Effect(
                EffectType.CARD_DISCARD,
                _CARD_DISCARD,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
