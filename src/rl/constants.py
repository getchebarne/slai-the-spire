"""
Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

ASCENSION_LEVEL = 1

# Skip trivial single-legal-action states in the engine (auto-advance). Used
# for training and eval — env-construction flag, not a tunable hyperparameter.
FAST_MODE = True

# Actor encoding limits
BLOCK_MAX_CHARACTER = 20
BLOCK_MAX_MONSTER = 20
HEALTH_MAX_CHARACTER = 70
HEALTH_MAX_MONSTER = 250  # From The Guardian at Ascension 9+

# Energy encoding
ENERGY_MIN = 0
ENERGY_MAX = 5


import slai as _slai


MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_MONSTERS = _slai.GameEnv.MAX_MONSTERS
MAX_SIZE_REWARD_CARDS = _slai.GameEnv.MAX_COMBAT_CARD_REWARD
MAX_SIZE_HAND = _slai.GameEnv.MAX_SIZE_HAND

# Encoder-side caps. slai's deck and draw/discard piles grow unboundedly;
# these are the maximum sizes the encoder pads to (overflow warns + drops the
# affected actions, see masks.py/card.py). Draw/discard must hold a full deck,
# so all three move together. Bump if Silent decks routinely exceed.
MAX_SIZE_DECK = 35
MAX_SIZE_DISC_PILE = 35
MAX_SIZE_DRAW_PILE = 35

# Exhausted cards stay few in Act 1; tokens are costly, so this cap is tighter.
MAX_SIZE_EXHAUST = 20

# Maximum number of relic-reward offers per RELIC_REWARD halt. Today slai
# emits at most 1 (post-Elite). Bump if that ever changes.
MAX_RELIC_REWARDS = 1

# Combat/Elite/chest rewards offer at most one potion; sized as a 1-slot
# sequence for representation uniformity with the relic reward.
MAX_POTION_REWARDS = 1

# Max owned relics encoded as entities. No duplicate relics, so the true
# ceiling is RelicName::COUNT; sized with headroom, bump as content grows.
MAX_RELICS = 16

# New-screen encoder/selection caps. Sized to current slai content with a
# little headroom; masks/encoders enumerate up to these. slai shops offer
# 7 cards / 3 relics / 3 potions; events up to ~9 options; discover ~3.
MAX_POTION_SLOTS = 3  # belt default (2 at A11+; extra slot masked empty)
MAX_SIZE_SHOP_CARDS = 8
MAX_SHOP_RELICS = 4
MAX_SHOP_POTIONS = 4
MAX_EVENT_OPTIONS = 12
MAX_SIZE_DISCOVER = 5

# Normalization caps for new scalar features
GOLD_CAP = 999
SHOP_PRICE_CAP = 400
EVENT_STATE_CAP = 10


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
    # Game constants
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "MAX_MONSTERS",
    "MAX_SIZE_REWARD_CARDS",
    "MAX_SIZE_DECK",
    "MAX_SIZE_DISC_PILE",
    "MAX_SIZE_DRAW_PILE",
    "MAX_SIZE_EXHAUST",
    "MAX_SIZE_HAND",
    "MAX_RELIC_REWARDS",
    "MAX_POTION_REWARDS",
    "MAX_RELICS",
    # Training flags + new-screen caps
    "FAST_MODE",
    "MAX_POTION_SLOTS",
    "MAX_SIZE_SHOP_CARDS",
    "MAX_SHOP_RELICS",
    "MAX_SHOP_POTIONS",
    "MAX_EVENT_OPTIONS",
    "MAX_SIZE_DISCOVER",
    "GOLD_CAP",
    "SHOP_PRICE_CAP",
    "EVENT_STATE_CAP",
]
