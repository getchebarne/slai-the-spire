from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.card import EntityCard


COST = 1
BLOCK = 5


def create_card_defend() -> EntityCard:
    return EntityCard(
        "Defend", COST, [Effect(EffectType.GAIN_BLOCK, BLOCK, EffectTargetType.CHARACTER)]
    )
