"""
Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

import slai as _slai


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
BLOCK_MAX_ENTITY = 40  # Spheric Guardian's opener; not derivable (templates expose no block)
# The catalog's widest monster HP roll (Champ 440 at Ascension 9+); the
# Character rides the same buckets far below
HEALTH_MAX_ENTITY = max(
    template.hp_range[1]
    for ascension in range(ASCENSION_LEVEL + 1)
    for template in _slai.get_monster_templates(ascension)
)


MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_EVENT_ROLL_CARDS = _slai.GameEnv.MAX_EVENT_ROLL_CARDS
MAX_EVENT_ROLL_POTIONS = _slai.GameEnv.MAX_EVENT_ROLL_POTIONS
MAX_EVENT_ROLL_RELICS = _slai.GameEnv.MAX_EVENT_ROLL_RELICS
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

# Relic-reward offers per RELIC_REWARD halt: boss rewards roll mutually
# exclusive relics (taking one discards the others).
MAX_RELIC_REWARDS = _slai.GameEnv.BOSS_RELIC_REWARD_COUNT

# Cauldron stages CAULDRON_POTION_COUNT potions as one reward batch; combat
# rewards offer at most 1.
MAX_POTION_REWARDS = _slai.GameEnv.CAULDRON_POTION_COUNT

# Max owned relics encoded as entities. No duplicate relics, so the true
# ceiling is RelicName::COUNT; sized with headroom, bump as content grows.
# Provisional two-act sizing (R4 harness).
MAX_RELICS = 24

# New-screen encoder/selection caps: engine slot counts + provisional headroom;
# masks/encoders enumerate up to these. Events run up to ~9 options today.
MAX_POTION_SLOTS = _slai.GameEnv.POTION_SLOTS_MAX  # unused slots masked empty
MAX_SIZE_SHOP_CARDS = _slai.GameEnv.SHOP_SLOTS_CARD_TOTAL + 1
MAX_SHOP_RELICS = _slai.GameEnv.SHOP_SLOTS_RELIC + 1
MAX_SHOP_POTIONS = _slai.GameEnv.SHOP_SLOTS_POTION + 1
MAX_EVENT_OPTIONS = 12
# Per-state effect-row pool. Ragged per entity (0-5 effects), so rows are packed
# per STATE with an owner segment rather than padded per entity: (B, 284, 5, D)
# would cost ~6x this. Measured live: mean 16.9 rows/state, max 35.
MAX_EFFECTS_STATE = 224
MAX_MODIFIER_ROWS_STATE = 64
MAX_SIZE_DISCOVER = _slai.GameEnv.DISCOVER_PICK_COUNT + 2

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
    "BLOCK_MAX_ENTITY",
    "HEALTH_MAX_ENTITY",
    # Game constants
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "MAX_EVENT_ROLL_CARDS",
    "MAX_EVENT_ROLL_POTIONS",
    "MAX_EVENT_ROLL_RELICS",
    "MAX_MONSTERS",
    "MAX_REWARD_CARD_BUNDLES",
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
    "MAX_MODIFIER_ROWS_STATE",
    "MAX_EVENT_OPTIONS",
    "MAX_SIZE_DISCOVER",
    "GOLD_CAP",
    "SHOP_PRICE_CAP",
    "EVENT_STATE_CAP",
]
