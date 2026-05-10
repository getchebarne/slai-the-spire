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
# Game-shape constants — pulled from slai where it owns them; encoder-side
# caps (deck / draw / disc piles) stay hardcoded since slai's piles are
# unbounded.
# =============================================================================

import slai as _slai

MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_MONSTERS = _slai.GameEnv.MAX_MONSTERS
MAX_SIZE_COMBAT_CARD_REWARD = _slai.GameEnv.MAX_COMBAT_CARD_REWARD
MAX_SIZE_HAND = _slai.GameEnv.MAX_SIZE_HAND

# Encoder-side caps. slai's deck and draw/discard piles grow unboundedly;
# these are the maximum sizes the encoder pads to. Bump if Silent decks
# routinely exceed.
MAX_SIZE_DECK = 50
MAX_SIZE_DISC_PILE = 50
MAX_SIZE_DRAW_PILE = 50

# Maximum number of relic-reward offers per RELIC_REWARD halt. Today slai
# emits at most 1 (post-Elite). Bump if that ever changes.
MAX_RELIC_REWARDS = 1


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
