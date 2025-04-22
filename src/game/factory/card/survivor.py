from src.game.combat.effect import Effect
from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
BLOCK = 8
CARD_DISCARD = 1


def create_card_survivor() -> EntityCard:
    return EntityCard(
        "Survivor",
        COST,
        [
            Effect(EffectType.BLOCK_GAIN, BLOCK, EffectTargetType.CHARACTER),
            Effect(
                EffectType.CARD_DISCARD,
                CARD_DISCARD,  # TODO: this should be part of the selection type
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )
