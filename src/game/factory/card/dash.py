from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types import CardUpgraded


_NAME = "Dash"
_COST = 2
_BLOCK = 10
_DAMAGE = 10
_BLOCK_PLUS = 13
_DAMAGE_PLUS = 13


@register_factory(_NAME)
def create_card_dash(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [
                Effect(EffectType.BLOCK_GAIN, _BLOCK_PLUS, EffectTargetType.CHARACTER),
                Effect(EffectType.DAMAGE_DEAL, _DAMAGE_PLUS, EffectTargetType.CARD_TARGET),
            ],
        )

    return EntityCard(
        _NAME,
        _COST,
        [
            Effect(EffectType.BLOCK_GAIN, _BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.DAMAGE_DEAL, _DAMAGE, EffectTargetType.CARD_TARGET),
        ],
    )
