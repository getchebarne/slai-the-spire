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


_NAME = "All Out Attack"
_COLOR = CardColor.GREEN
_COST = 1
_DAMAGE = 10
_DAMAGE_PLUS = 14
_DISCARD = 1
_RARITY = CardRarity.UNCOMMON
_TYPE = CardType.ATTACK


@register_factory(_NAME)
def create_card_all_out_attack(upgraded: CardUpgraded) -> EntityCard:
    if upgraded:
        return EntityCard(
            f"{_NAME}+",
            _COLOR,
            _TYPE,
            _COST,
            _RARITY,
            [
                Effect(EffectType.DAMAGE_DEAL, _DAMAGE_PLUS, EffectTargetType.MONSTER),
                Effect(
                    EffectType.CARD_DISCARD,
                    _DISCARD,
                    target_type=EffectTargetType.CARD_IN_HAND,
                    selection_type=EffectSelectionType.RANDOM,
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
            Effect(EffectType.DAMAGE_DEAL, _DAMAGE, EffectTargetType.MONSTER),
            Effect(
                EffectType.CARD_DISCARD,
                _DISCARD,
                target_type=EffectTargetType.CARD_IN_HAND,
                selection_type=EffectSelectionType.RANDOM,
            ),
        ],
    )
