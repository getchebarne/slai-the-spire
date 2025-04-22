from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 2
BLOCK = 11
WEAK = 2


def create_card_leg_sweep() -> EntityCard:

    return EntityCard(
        "Leg Sweep",
        COST,
        [
            Effect(EffectType.BLOCK_GAIN, BLOCK, EffectTargetType.CHARACTER),
            Effect(EffectType.MODIFIER_WEAK_GAIN, WEAK, EffectTargetType.CARD_TARGET),
        ],
    )
