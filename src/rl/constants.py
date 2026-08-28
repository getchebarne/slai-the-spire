"""
Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

ASCENSION_LEVEL = 20

# Skip trivial single-legal-action states in the engine (auto-advance). Used
# for training and eval — env-construction flag, not a tunable hyperparameter.
# Disabled 2026-07: holding a potion makes PotionDiscard legal everywhere, which
# un-trivializes otherwise-skipped states — a pure surfaced-step inflation that,
# under per-step discounting, taxes potion-holding and biases toward early discards.
FAST_MODE = False

# Actor encoding limits. Monster and character share ENTITY caps: both encoders
# must build a byte-identical health/block block so EntityProjector can share
# its first-layer family weights (encoder import-time asserts enforce widths).
BLOCK_MAX_ENTITY = 40  # Spheric Guardian starts at 40 and lives there (Barricade)
HEALTH_MAX_ENTITY = 450  # From Champ at Ascension 9+ (440)
BLOCK_MAX_CHARACTER = 20  # DEAD: superseded by BLOCK_MAX_ENTITY; kept for reference
HEALTH_MAX_CHARACTER = (
    100  # DEAD: superseded by HEALTH_MAX_ENTITY (was: Silent 70 base + max-HP events, R4)
)

# Energy encoding
ENERGY_MIN = 0
ENERGY_MAX = 5


import slai as _slai


MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_MONSTERS = _slai.GameEnv.MAX_MONSTERS
MAX_SIZE_HAND = _slai.GameEnv.MAX_SIZE_HAND

# Reward cards arrive as bundles (Reward.cards: list[list[Card]]): up to
# MAX_REWARD_CARD_BUNDLES combat bundles of <= MAX_COMBAT_CARD_REWARD cards
# (Orrery stages 5), or The Library's single MAX_CARD_REWARD_ROLL-card bundle.
# Each bundle owns its own CARD_REWARD_<b> slice — bundle 0 wide enough for
# The Library, the rest hold combat rolls — so slot <-> (bundle, card) is
# static, via the tables below
MAX_REWARD_CARD_BUNDLES = _slai.GameEnv.MAX_REWARD_CARD_BUNDLES
REWARD_REGION_WIDTHS = (_slai.GameEnv.MAX_CARD_REWARD_ROLL,) + (
    _slai.GameEnv.MAX_COMBAT_CARD_REWARD,
) * (MAX_REWARD_CARD_BUNDLES - 1)
REWARD_REGION_OFFSETS = tuple(
    sum(REWARD_REGION_WIDTHS[:bundle]) for bundle in range(MAX_REWARD_CARD_BUNDLES)
)
REWARD_SLOT_TO_BUNDLE = tuple(
    bundle for bundle, width in enumerate(REWARD_REGION_WIDTHS) for _ in range(width)
)
REWARD_SLOT_TO_CARD = tuple(idx for width in REWARD_REGION_WIDTHS for idx in range(width))
MAX_REWARD_CARD_TOKENS = sum(REWARD_REGION_WIDTHS)

# Cards exiled by Bronze Orb StasisSteal; the engine slot-parallels them to the
# monster arena ([Option; MAX_MONSTERS]), so this cap is exact, not headroom.
MAX_SIZE_STASIS = MAX_MONSTERS

# Encoder-side caps. slai's deck and draw/discard piles grow unboundedly;
# these are the maximum sizes the encoder pads to (overflow warns + drops the
# affected actions, see masks.py/card.py). Draw/discard must hold a full deck,
# so all three move together. Provisional two-act sizing (R4 harness confirms).
MAX_SIZE_DECK = 45
MAX_SIZE_DISC_PILE = 45
MAX_SIZE_DRAW_PILE = 45

# Tighter than the deck caps (tokens are costly), but two acts of exhaust
# cycling run deeper than Act 1 did. R6's on-policy smoke hit 32 (vs 24 from
# random play), so sized with real headroom; observational-only overflow.
MAX_SIZE_EXHAUST = 40

# Maximum number of relic-reward offers per RELIC_REWARD halt: boss rewards
# roll 3 mutually exclusive relics (taking one discards the others).
MAX_RELIC_REWARDS = 3

# Cauldron stages CAULDRON_POTION_COUNT potions as one reward batch; combat
# rewards offer at most 1.
MAX_POTION_REWARDS = _slai.GameEnv.CAULDRON_POTION_COUNT

# Max owned relics encoded as entities. No duplicate relics, so the true
# ceiling is RelicName::COUNT; sized with headroom, bump as content grows.
# Provisional two-act sizing (R4 harness).
MAX_RELICS = 24

# New-screen encoder/selection caps. Sized to current slai content with a
# little headroom; masks/encoders enumerate up to these. slai shops offer
# 7 cards / 3 relics / 3 potions; events up to ~9 options; discover ~3.
MAX_POTION_SLOTS = 5  # belt 3 (2 at A11+) + Potion Belt's 2; unused slots masked empty
MAX_SIZE_SHOP_CARDS = 8
MAX_SHOP_RELICS = 4
MAX_SHOP_POTIONS = 4
MAX_EVENT_OPTIONS = 12
# Per-state effect-row pool. Ragged per entity (0-5 effects), so rows are packed
# per STATE with an owner segment rather than padded per entity: (B, 284, 5, D)
# would cost ~6x this. Measured live: mean 16.9 rows/state, max 35.
MAX_EFFECTS_STATE = 224
MAX_SIZE_DISCOVER = 5

# Normalization caps for new scalar features
GOLD_CAP = 999
# Exact pre-discount ceiling, derived from the engine's own pricing formula
# (relics are the most expensive slot); discounts only lower prices.
SHOP_PRICE_CAP = max(
    _slai.SHOP_PRICE_CARD_MAX, _slai.SHOP_PRICE_RELIC_MAX, _slai.SHOP_PRICE_POTION_MAX
)
EVENT_STATE_CAP = 10


__all__ = [
    # Training
    "ASCENSION_LEVEL",
    # Encoding limits
    "BLOCK_MAX_CHARACTER",
    "BLOCK_MAX_ENTITY",
    "HEALTH_MAX_CHARACTER",
    "HEALTH_MAX_ENTITY",
    "ENERGY_MIN",
    "ENERGY_MAX",
    # Game constants
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "MAX_MONSTERS",
    "MAX_REWARD_CARD_BUNDLES",
    "MAX_REWARD_CARD_TOKENS",
    "REWARD_REGION_WIDTHS",
    "REWARD_REGION_OFFSETS",
    "REWARD_SLOT_TO_BUNDLE",
    "REWARD_SLOT_TO_CARD",
    "MAX_SIZE_STASIS",
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
    "MAX_EFFECTS_STATE",
    "MAX_EVENT_OPTIONS",
    "MAX_SIZE_DISCOVER",
    "GOLD_CAP",
    "SHOP_PRICE_CAP",
    "EVENT_STATE_CAP",
]
