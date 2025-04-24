from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types import CardUpgraded


_NAME = "Leg Sweep"
_COST = 2
_BLOCK = 11
_BLOCK_PLUS = 14
_WEAK = 2
_WEAK_PLUS = 3


@register_factory(_NAME)
def create_card_leg_sweep(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK_PLUS, EffectTargetType.CARD_TARGET),
            ],
        )

    return EntityCard(
        _NAME,
        _COST,
        [
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK, EffectTargetType.CARD_TARGET),
        ],
    )
