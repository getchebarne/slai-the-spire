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

# Start runs with Neow's blessing (choice states + bonuses). Enabled 2026-08-30
# for EFFECT-II; EFFECT-I and earlier trained without it.
NEOW = True

# Actor encoding limits. Monster and character share ENTITY caps: health_block.py
# builds the one health/block row both feed EntityProjector's shared family block.
# Every Monster template across every ascension, for the catalog folds below
_MONSTER_TEMPLATES = [
    template
    for ascension in range(ASCENSION_LEVEL + 1)
    for template in _slai.get_monster_templates(ascension)
]
_MONSTER_MOVES = [move for template in _MONSTER_TEMPLATES for move in template.moves]

# Character max-health normalization clamp (health grows via events/relics with no
# engine ceiling); also the scale Hexaghost's Divider is bridged from
CHARACTER_HEALTH_CAP = 200

# The catalog's widest single block source (Spheric Guardian's opener); Character
# block accumulation clamps to the same scale
BLOCK_MAX_ENTITY = max(
    value
    for template in _MONSTER_TEMPLATES
    for value in (
        template.block_start,
        *(
            effect.amount
            for move in template.moves
            for effect in move.effects
            if isinstance(effect, _slai.EffectBlockGain)
        ),
    )
)

# The catalog's widest monster HP roll (Champ 440 at Ascension 9+); the
# Character rides the same buckets far below
HEALTH_MAX_ENTITY = max(template.hp_range[1] for template in _MONSTER_TEMPLATES)

# Widest raw intent hits. Hexaghost's Divider template carries a placeholder 0
# (damage rolls at selection: character health / DIVISOR + 1), so it is bridged
# from CHARACTER_HEALTH_CAP; vuln scaling is applied by the encoder caps
_DIVIDER_HIT_MAX = CHARACTER_HEALTH_CAP // _slai.GameEnv.HEXAGHOST_DIVIDER_DIVISOR + 1
INTENT_DAMAGE_HIT_MAX = max(
    max(move.damage for move in _MONSTER_MOVES if move.damage is not None),
    _DIVIDER_HIT_MAX,
)
# Book of Stabbing's Multi-Stab template carries a placeholder 0 hits (the count
# locks at selection and grows with every use), so it is bridged at the
# engine's former 20-hit ceiling
MULTI_STAB_INSTANCES_MAX = 20
_MULTI_STAB_HIT_MAX = max(
    move.damage for move in _MONSTER_MOVES if move.name == "Multi-Stab" and move.damage is not None
)
INTENT_DAMAGE_TOTAL_MAX = max(
    max(move.damage * move.instances for move in _MONSTER_MOVES if move.damage is not None),
    _DIVIDER_HIT_MAX * _slai.GameEnv.HEXAGHOST_DIVIDER_HITS,
    _MULTI_STAB_HIT_MAX * MULTI_STAB_INSTANCES_MAX,
)
INTENT_INSTANCES_MAX = max(
    max(move.instances for move in _MONSTER_MOVES if move.instances is not None),
    MULTI_STAB_INSTANCES_MAX,
)
# Thievery steals its stacks per landed attack (Looter/Mugger spawn modifier)
# Widest spawn-rolled moveset (Book of Stabbing's Multi-Stab ramp); move-index OHE width
MONSTER_MOVESET_MAX = max(len(template.moves) for template in _MONSTER_TEMPLATES)

GOLD_STEAL_MAX = max(
    spawn.stacks_max
    for template in _MONSTER_TEMPLATES
    for spawn in template.modifier_spawns
    if spawn.kind is _slai.ModifierKind.Thievery
)
FACTOR_VULN = _slai.GameEnv.FACTOR_VULN


MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_SIZE_CARD_PILE_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_CARD_PILE_EVENT_ROLL
MAX_SIZE_POTION_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_POTION_EVENT_ROLL
MAX_SIZE_RELIC_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_RELIC_EVENT_ROLL
MAX_MONSTERS = _slai.GameEnv.MAX_MONSTERS
MAX_SIZE_CARD_PILE_HAND = _slai.GameEnv.MAX_SIZE_CARD_PILE_HAND

# Reward cards arrive as bundles (Reward.cards: list[list[Card]]): up to
# MAX_REWARD_CARD_BUNDLES of them (Orrery stages 5). Each bundle is its own
# CARD_REWARD_<b> kind, so a reward token's kind names its bundle
MAX_REWARD_CARD_BUNDLES = _slai.GameEnv.MAX_REWARD_CARD_BUNDLES

# Size ceiling for the master deck, used to normalize deck-scale counts (deck
# size, halt picks over the deck and the combat piles it feeds). A trained
# agent's deep runs hit 46 (EFFECT-II iter ~4040)
MAX_SIZE_CARD_PILE_DECK = 55

# Relic-reward offers per RELIC_REWARD halt: boss rewards roll mutually
# exclusive relics (taking one discards the others).
MAX_RELIC_REWARDS = _slai.GameEnv.BOSS_RELIC_REWARD_COUNT

# Cauldron stages CAULDRON_POTION_COUNT potions as one reward batch; combat
# rewards offer at most 1.
MAX_POTION_REWARDS = _slai.GameEnv.CAULDRON_POTION_COUNT

# Engine slot counts + provisional headroom. Events run up to ~9 options today.
MAX_POTION_SLOTS = _slai.GameEnv.POTION_SLOTS_MAX  # belt capacity; the FFI list is dense
MAX_SIZE_SHOP_CARDS = _slai.GameEnv.SHOP_SLOTS_CARD_TOTAL + 1
MAX_SHOP_RELICS = _slai.GameEnv.SHOP_SLOTS_RELIC + 1
MAX_SHOP_POTIONS = _slai.GameEnv.SHOP_SLOTS_POTION + 1
MAX_EVENT_OPTIONS = 12
# Per-state effect-row pool. Ragged per entity (0-5 effects), so rows are packed
# per STATE with an owner segment rather than padded per entity: (B, 284, 5, D)
# would cost ~6x this. Measured under EFFECT-II@4100 (trained agent, deep act-2
# runs) with moveset variants deduped: mean 52, p99 109, max 116
MAX_EFFECTS_STATE = 224
MAX_MODIFIER_ROWS_STATE = 64
MAX_SIZE_CARD_PILE_DISCOVER = _slai.GameEnv.DISCOVER_PICK_COUNT + 2

# Per-collection caps for the collections the engine does not bound itself. Each
# encoder slices its pile to the cap under that kind's `case` arm; a type's block
# is as wide as the batch's longest packed sequence, and only the rollout
# transport (train.py shared memory) is sized to a type's summed caps.
# Draw/discard must hold a full deck, so they move with MAX_SIZE_CARD_PILE_DECK; a trained
# agent's deep runs hit 46 cards (EFFECT-II iter ~4040). Exhaust: two acts of
# exhaust cycling ran 41-42 under EFFECT-II. Stasis is slot-parallel to the
# monster arena in the engine, so that cap is exact.
MAX_SIZE_CARD_PILE_DRAW = 55
MAX_SIZE_CARD_PILE_DISCARD = 55
MAX_SIZE_CARD_PILE_EXHAUST = 45
MAX_SIZE_CARD_PILE_STASIS = MAX_MONSTERS
# Staged picks of an open multi-pick halt: at most a hand's worth
MAX_PENDING_SELECTED = MAX_SIZE_CARD_PILE_HAND
# Queued plays (replays, top-of-draw plays, end-of-turn self-plays) and the
# nightmare pile: at most a hand's worth; past it they warn and drop
MAX_SIZE_CARD_PLAY_QUEUE = MAX_SIZE_CARD_PILE_HAND
MAX_SIZE_CARD_PILE_NIGHTMARE = MAX_SIZE_CARD_PILE_HAND
# Owned relics: no duplicates, so the true ceiling is the catalog; sized with headroom
MAX_RELICS = 24
# Reward bundle widths: bundle 0 must hold The Library's single wide roll, the
# rest hold combat rolls
MAX_CARD_REWARD_ROLL = _slai.GameEnv.MAX_CARD_REWARD_ROLL
MAX_COMBAT_CARD_REWARD = _slai.GameEnv.MAX_COMBAT_CARD_REWARD

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
    "CHARACTER_HEALTH_CAP",
    "INTENT_DAMAGE_HIT_MAX",
    "INTENT_DAMAGE_TOTAL_MAX",
    "INTENT_INSTANCES_MAX",
    "MULTI_STAB_INSTANCES_MAX",
    "GOLD_STEAL_MAX",
    "MONSTER_MOVESET_MAX",
    "FACTOR_VULN",
    # Game constants
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "MAX_SIZE_CARD_PILE_EVENT_ROLL",
    "MAX_SIZE_POTION_EVENT_ROLL",
    "MAX_SIZE_RELIC_EVENT_ROLL",
    "MAX_MONSTERS",
    "MAX_REWARD_CARD_BUNDLES",
    "MAX_SIZE_CARD_PILE_DECK",
    "MAX_SIZE_CARD_PILE_HAND",
    "MAX_RELIC_REWARDS",
    "MAX_POTION_REWARDS",
    # Per-collection caps
    "MAX_SIZE_CARD_PILE_DRAW",
    "MAX_SIZE_CARD_PILE_DISCARD",
    "MAX_SIZE_CARD_PILE_EXHAUST",
    "MAX_SIZE_CARD_PILE_STASIS",
    "MAX_PENDING_SELECTED",
    "MAX_SIZE_CARD_PLAY_QUEUE",
    "MAX_SIZE_CARD_PILE_NIGHTMARE",
    "MAX_RELICS",
    "MAX_CARD_REWARD_ROLL",
    "MAX_COMBAT_CARD_REWARD",
    # Training flags + new-screen caps
    "FAST_MODE",
    "NEOW",
    "MAX_POTION_SLOTS",
    "MAX_SIZE_SHOP_CARDS",
    "MAX_SHOP_RELICS",
    "MAX_SHOP_POTIONS",
    "MAX_EFFECTS_STATE",
    "MAX_MODIFIER_ROWS_STATE",
    "MAX_EVENT_OPTIONS",
    "MAX_SIZE_CARD_PILE_DISCOVER",
    "GOLD_CAP",
    "SHOP_PRICE_CAP",
    "EVENT_STATE_CAP",
]
