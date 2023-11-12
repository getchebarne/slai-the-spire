from game.pipeline.steps.apply_str import ApplyStrength
from game.pipeline.steps.deal_damage import DealDamage
from game.pipeline.steps.gain_block import GainBlock
from game.pipeline.steps.gain_str import GainStrength

# TODO: this is a temporary solution to easily set the steps' priority
# while the code is developed. Once all steps are implemented, this will be
# removed

# Block
STEP_ORDER_BLOCK = [
    GainBlock,
]

# Damage
STEP_ORDER_DAMAGE = [
    ApplyStrength,
    DealDamage,
]

# Modifiers
STEP_ORDER_MODIFIERS = [
    GainStrength,
]

# All
STEP_ORDER = STEP_ORDER_BLOCK + STEP_ORDER_DAMAGE + STEP_ORDER_MODIFIERS
