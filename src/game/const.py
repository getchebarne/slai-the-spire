from src.game.core.effect import EffectType


# Maximum number of monsters
MAX_MONSTERS = 2

# Hand, draw pile, and discard pile maximum sizes
MAX_SIZE_HAND = 7
MAX_SIZE_DRAW_PILE = 16
MAX_SIZE_DISC_PILE = 16

# Types of effects that appear in cards
EFFECT_TYPE_CARD = [
    EffectType.DAMAGE_DEAL_PHYSICAL,
    EffectType.BLOCK_GAIN,
    EffectType.CARD_DISCARD,
    EffectType.MODIFIER_WEAK_GAIN,
    EffectType.CARD_DRAW,
]
