from src.game.combat.effect import EffectType


# Hand, draw pile, and discard pile maximum sizes
MAX_SIZE_HAND = 7
MAX_SIZE_DRAW_PILE = 16
MAX_SIZE_DISC_PILE = 16

# Types of effects that appear in cards
EFFECT_TYPE_CARD = [
    EffectType.DEAL_DAMAGE,
    EffectType.GAIN_BLOCK,
    EffectType.DISCARD,
    EffectType.GAIN_WEAK,
    EffectType.DRAW_CARD,
]
