from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.factory.lib import register_factory
from src.game.types_ import CardUpgraded


_NAME = "Strike"
_COST = 1
_DAMAGE = 6
_DAMAGE_PLUS = 9


@register_factory(_NAME)
def create_card_strike(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COST,
            [Effect(EffectType.DAMAGE_DEAL, _DAMAGE_PLUS, EffectTargetType.CARD_TARGET)],
        )

    return EntityCard(
        _NAME,
        _COST,
        [Effect(EffectType.DAMAGE_DEAL, _DAMAGE, EffectTargetType.CARD_TARGET)],
    )
