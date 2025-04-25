from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Neutralize"
_COST = 0
_DAMAGE = 3
_DAMAGE_PLUS = 4
_WEAK = 1
_WEAK_PLUS = 2


@register_factory(_NAME)
def create_card_neutralize(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [
                Effect(EffectType.DAMAGE_DEAL, _DAMAGE_PLUS, EffectTargetType.CARD_TARGET),
                Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK_PLUS, EffectTargetType.CARD_TARGET),
            ],
        )

    return EntityCard(
        _NAME,
        _COST,
        [
            Effect(EffectType.DAMAGE_DEAL, _DAMAGE, EffectTargetType.CARD_TARGET),
            Effect(EffectType.MODIFIER_WEAK_GAIN, _WEAK, EffectTargetType.CARD_TARGET),
        ],
    )
