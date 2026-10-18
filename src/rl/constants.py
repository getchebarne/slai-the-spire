"""
Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

import slai as _slai


# The top Ascension, which GameEnv doesn't expose; it sizes the ascension thermometer
# (a checkpoint shape) and every catalog fold
ASCENSION_LEVEL = 20

# Skip single-legal-action states in the engine (auto-advance), in training and eval
# Off: a held Potion keeps PotionDiscard legal, so the extra steps it surfaces tax holding one
FAST_MODE = False

# Start runs with Neow's blessing (choice states + bonuses)
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

# Widest raw intent hit; Vulnerable scaling is the encoder caps' job. Hexaghost's Divider
# template holds a placeholder 0 (it rolls at selection): bridged from CHARACTER_HEALTH_CAP
_DIVIDER_HIT_MAX = CHARACTER_HEALTH_CAP // _slai.GameEnv.HEXAGHOST_DIVIDER_DIVISOR + 1
INTENT_DAMAGE_HIT_MAX = max(
    max(move.damage for move in _MONSTER_MOVES if move.damage is not None),
    _DIVIDER_HIT_MAX,
)

# Book of Stabbing's Multi-Stab template holds a placeholder 0 hits (the count grows with each
# use), so it is bridged at the engine's former 20-hit ceiling
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

# Widest moveset in the catalog: the move-index OHE width
MONSTER_MOVESET_MAX = max(len(template.moves) for template in _MONSTER_TEMPLATES)

# Thievery steals its stacks per landed attack (Looter/Mugger spawn modifier)
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

# Most card bundles one Reward holds (Orrery stages 5); each bundle is its own
# CARD_REWARD_<b> slice kind, so a reward token's slice kind names its bundle
MAX_REWARD_CARD_BUNDLES = _slai.GameEnv.MAX_REWARD_CARD_BUNDLES

# Size ceiling for the master deck, used to normalize deck-scale counts (deck
# size, halt selections over the deck and the combat piles it feeds)
MAX_SIZE_CARD_PILE_DECK = 55

# Relics offered per reward: boss rewards roll mutually
# exclusive relics (taking one discards the others).
MAX_RELIC_REWARDS = _slai.GameEnv.BOSS_RELIC_REWARD_COUNT

# Cauldron stages CAULDRON_POTION_COUNT potions as one reward batch; combat
# rewards offer at most 1.
MAX_POTION_REWARDS = _slai.GameEnv.CAULDRON_POTION_COUNT

# Engine slot counts + provisional headroom
MAX_POTION_SLOTS = _slai.GameEnv.POTION_SLOTS_MAX  # belt capacity; the FFI list is dense
MAX_SIZE_SHOP_CARDS = _slai.GameEnv.SHOP_SLOTS_CARD_TOTAL + 1
MAX_SHOP_RELICS = _slai.GameEnv.SHOP_SLOTS_RELIC + 1
MAX_SHOP_POTIONS = _slai.GameEnv.SHOP_SLOTS_POTION + 1
MAX_EVENT_OPTIONS = 12
MAX_SIZE_CARD_PILE_DISCOVER = _slai.GameEnv.DISCOVER_ROLL_COUNT + 2

# Per-state effect-row pool, packed per STATE with an owner segment rather than padded per
# entity, since entities are ragged (a Monster brings its whole moveset)
MAX_EFFECTS_STATE = 224
MAX_MODIFIER_ROWS_STATE = 64

# Caps for the piles the engine doesn't bound: each encoder slices its pile to the cap. Draw
# and discard hold a full deck; stasis is slot-parallel to the monsters, so its cap is exact
MAX_SIZE_CARD_PILE_DRAW = MAX_SIZE_CARD_PILE_DECK
MAX_SIZE_CARD_PILE_DISCARD = MAX_SIZE_CARD_PILE_DECK
MAX_SIZE_CARD_PILE_EXHAUST = 45
MAX_SIZE_CARD_PILE_STASIS = MAX_MONSTERS

# Staged selections of an open multi-selection halt: at most a hand's worth
MAX_SIZE_CARD_PILE_SELECTED = MAX_SIZE_CARD_PILE_HAND

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

# Normalization caps for scalar features; SHOP_PRICE_CAP is the exact pre-discount
# ceiling from the engine's pricing (discounts only lower prices)
SHOP_PRICE_CAP = max(
    _slai.SHOP_PRICE_CARD_MAX, _slai.SHOP_PRICE_RELIC_MAX, _slai.SHOP_PRICE_POTION_MAX
)
EVENT_STAGE_CAP = 10
