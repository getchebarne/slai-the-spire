from src.game.combat.effect import Effect
from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
BLOCK = 8
DISCARD = 1


def create_card_survivor() -> EntityCard:
    return EntityCard(
        "Survivor",
        COST,
        [
            Effect(EffectType.GAIN_BLOCK, BLOCK, EffectTargetType.CHARACTER),
            Effect(
                EffectType.DISCARD,
                DISCARD,  # TODO: this should be part of the selection type
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
