"""
Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

# =============================================================================
# Training
# =============================================================================

ASCENSION_LEVEL = 1

# =============================================================================
# Encoding Limits
# =============================================================================

# Actor encoding limits
BLOCK_MAX_CHARACTER = 20
BLOCK_MAX_MONSTER = 20
HEALTH_MAX_CHARACTER = 70
HEALTH_MAX_MONSTER = 250  # From The Guardian at Ascension 9+

# Energy encoding
ENERGY_MIN = 0
ENERGY_MAX = 5

# Modifier stack limits (for normalization)
MODIFIER_STACKS_MAX = {
    "STRENGTH": 20,
    "WEAK": 5,
    "MODE_SHIFT": 60,
    "RITUAL": 20,
    "SHARP_HIDE": 3,
    "SPORE_CLOUD": 2,
    "VULNERABLE": 4,
    "ACCURACY": 16,
    "NEXT_TURN_BLOCK": 20,
    "NEXT_TURN_ENERGY": 5,
    "BLUR": 5,
    "DEXTERITY": 12,
    "INFINITE_BLADES": 5,
    "AFTER_IMAGE": 3,
    "PHANTASMAL": 2,
    "DOUBLE_DAMAGE": 1,
    "THOUSAND_CUTS": 4,
    "BURST": 4,
}

# =============================================================================
# Game-shape constants (mirror slai's Rust-side `consts.rs`; slai does not
# expose these via the Python API today, so they're hardcoded here).
# =============================================================================

MAP_HEIGHT = 15
MAP_WIDTH = 7
MAX_MONSTERS = 2  # RL-side cap; slai supports up to 5. See migration plan.
MAX_SIZE_COMBAT_CARD_REWARD = 3
MAX_SIZE_DECK = 25  # dynamic in slai; this is an encoder-side cap
MAX_SIZE_DISC_PILE = 25
MAX_SIZE_DRAW_PILE = 25
MAX_SIZE_HAND = 10


__all__ = [
    # Training
    "ASCENSION_LEVEL",
    # Encoding limits
    "BLOCK_MAX_CHARACTER",
    "BLOCK_MAX_MONSTER",
    "HEALTH_MAX_CHARACTER",
    "HEALTH_MAX_MONSTER",
    "ENERGY_MIN",
    "ENERGY_MAX",
    "MODIFIER_STACKS_MAX",
    # Game constants
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "MAX_MONSTERS",
    "MAX_SIZE_COMBAT_CARD_REWARD",
    "MAX_SIZE_DECK",
    "MAX_SIZE_DISC_PILE",
    "MAX_SIZE_DRAW_PILE",
    "MAX_SIZE_HAND",
]
