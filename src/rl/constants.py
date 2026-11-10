"""Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

import slai as _slai

from src.rl.types import TokenKind


# The policy's action types are exactly the engine's
ACTION_TYPE_BY_INT = _slai.ActionType.members()
NUM_ACTION_TYPES = len(ACTION_TYPE_BY_INT)

# The top Ascension, which GameEnv doesn't expose; it sizes the ascension thermometer
# (a checkpoint shape) and every catalog fold
ASCENSION_LEVEL = 20

# Skip single-legal-action states in the engine (auto-advance), in training and eval
# Off: a held Potion keeps PotionDiscard legal, so the extra steps it surfaces tax holding one
FAST_MODE = False

# Start runs with Neow's blessing (choice states + bonuses)
NEOW = True

# Actor encoding limits. Monster and Character share ENTITY caps: health_block.py
# writes the health/block columns both share.

# Every Monster template across every ascension, for the catalog folds below
_MONSTER_TEMPLATES = [
    template
    for ascension in range(ASCENSION_LEVEL + 1)
    for template in _slai.get_monster_templates(ascension)
]
_MONSTER_MOVES = [move for template in _MONSTER_TEMPLATES for move in template.moves]

# Character max-health normalization clamp (health grows via Events/relics with no
# engine ceiling); also the scale Hexaghost's Divider is bridged from
CHARACTER_HEALTH_CAP = 200

# The catalog's widest single block source (the Spire Shield's A18 block); Character
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

# The catalog's widest Monster HP roll (the Transient's 999); the
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

# Thievery steals its stacks per landed attack (Looter/Mugger spawn Modifier)
_GOLD_STEAL_MAX = max(
    spawn.stacks_max
    for template in _MONSTER_TEMPLATES
    for spawn in template.modifier_spawns
    if spawn.kind is _slai.ModifierKind.Thievery
)

# Thievery accumulation: Mug, Mug, Lunge — every attack before the Escape turn steals
GOLD_STOLEN_MAX = 3 * _GOLD_STEAL_MAX

# Invincible caps the HP lost per turn at its stacks (the Corrupt Heart's spawn Modifier), which
# refill ahead of their holder's Poison tick
INVINCIBLE_STACKS_OF_ASCENSION = {
    ascension: max(
        spawn.stacks_max
        for template in _slai.get_monster_templates(ascension)
        for spawn in template.modifier_spawns
        if spawn.kind is _slai.ModifierKind.Invincible
    )
    for ascension in range(ASCENSION_LEVEL + 1)
}
INVINCIBLE_STACKS_MAX = max(INVINCIBLE_STACKS_OF_ASCENSION.values())

FACTOR_VULN = _slai.GameEnv.FACTOR_VULN


MAP_HEIGHT = _slai.GameEnv.MAP_HEIGHT
MAP_WIDTH = _slai.GameEnv.MAP_WIDTH
MAX_SIZE_CARD_PILE_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_CARD_PILE_EVENT_ROLL
MAX_SIZE_POTION_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_POTION_EVENT_ROLL
MAX_SIZE_RELIC_EVENT_ROLL = _slai.GameEnv.MAX_SIZE_RELIC_EVENT_ROLL
MAX_SIZE_CARD_MATCH_PAIRS = _slai.GameEnv.MATCH_AND_KEEP_ATTEMPTS  # One pair per attempt at most
MAX_MONSTERS = _slai.GameEnv.MAX_MONSTERS
MAX_SIZE_CARD_PILE_HAND = _slai.GameEnv.MAX_SIZE_CARD_PILE_HAND

# Most Card bundles one Reward holds (Orrery stages 5); each bundle is its own
# CARD_REWARD_<b> token kind, so a Reward token's token kind names its bundle
MAX_REWARD_CARD_BUNDLES = _slai.GameEnv.MAX_REWARD_CARD_BUNDLES

# Size ceiling for the master deck, used to normalize deck-scale counts (deck
# size, halt selections over the deck and the Combat piles it feeds)
MAX_SIZE_CARD_PILE_DECK = 80

# Relics offered per Reward: boss Rewards roll mutually
# exclusive Relics (taking one discards the others).
MAX_RELIC_REWARDS = _slai.GameEnv.BOSS_RELIC_REWARD_COUNT

# Cauldron stages CAULDRON_POTION_COUNT Potions as one Reward batch; Combat
# Rewards offer at most 1.
MAX_POTION_REWARDS = _slai.GameEnv.CAULDRON_POTION_COUNT

# Engine slot counts + provisional headroom
MAX_POTION_SLOTS = _slai.GameEnv.POTION_SLOTS_MAX  # Belt capacity; the FFI list is dense
MAX_SIZE_SHOP_CARDS = _slai.GameEnv.SHOP_SLOTS_CARD_TOTAL + 1
MAX_SHOP_RELICS = _slai.GameEnv.SHOP_SLOTS_RELIC + 1
MAX_SHOP_POTIONS = _slai.GameEnv.SHOP_SLOTS_POTION + 1
MAX_EVENT_OPTIONS = 12
MAX_SIZE_CARD_PILE_DISCOVER = _slai.GameEnv.DISCOVER_ROLL_COUNT + 2

# Per-state caps on the Effect and Modifier rows, per state rather than per Entity since
# Entities are ragged (a Monster brings its whole moveset); they size the workers' shared step
MAX_EFFECTS = 224
MAX_MODIFIERS = 64

# Caps for the piles the engine doesn't bound: each encoder slices its pile to the cap. Draw
# and discard hold a full deck; stasis is slot-parallel to the Monsters, so its cap is exact
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

# Owned Relics: no duplicates, so the true ceiling is the catalog; sized with headroom
MAX_RELICS = 24

# Reward bundle widths: bundle 0 must hold The Library's single wide roll, the
# rest hold Combat rolls
MAX_CARD_REWARD_ROLL = _slai.GameEnv.MAX_CARD_REWARD_ROLL
MAX_COMBAT_CARD_REWARD = _slai.GameEnv.MAX_COMBAT_CARD_REWARD

# Normalization caps for scalar features; SHOP_PRICE_CAP is the exact pre-discount
# ceiling from the engine's pricing (discounts only lower prices)
SHOP_PRICE_CAP = max(
    _slai.SHOP_PRICE_CARD_MAX, _slai.SHOP_PRICE_RELIC_MAX, _slai.SHOP_PRICE_POTION_MAX
)
EVENT_STAGE_CAP = 10


TOKEN_KINDS_CARD_REWARDS = (  # One token kind per Reward bundle (tokens.py owns their packing)
    TokenKind.CARD_REWARD_0,
    TokenKind.CARD_REWARD_1,
    TokenKind.CARD_REWARD_2,
    TokenKind.CARD_REWARD_3,
    TokenKind.CARD_REWARD_4,
)

# Pack order within the Card Entity kind: decision pools first, observational piles last
TOKEN_KINDS_CARD = (
    TokenKind.CARD_PILE_HAND,
    TokenKind.CARD_PILE_DISCOVER,
    TokenKind.CARD_PILE_SELECTED,
    *TOKEN_KINDS_CARD_REWARDS,
    TokenKind.CARD_SHOP,
    TokenKind.CARD_PILE_EVENT_ROLL,
    TokenKind.CARD_MATCH_FLIPPED,
    TokenKind.CARD_MATCH_PAIRS,
    TokenKind.CARD_PILE_STASIS,
    TokenKind.CARD_PLAY_QUEUE,
    TokenKind.CARD_PILE_NIGHTMARE,
    TokenKind.CARD_PILE_DRAW,
    TokenKind.CARD_PILE_DISCARD,
    TokenKind.CARD_PILE_EXHAUST,
    TokenKind.CARD_PILE_DECK,
)

# Pack order within the Relic Entity kind: decision pools first, the owned Relics last
TOKEN_KINDS_RELIC = (
    TokenKind.RELIC_REWARD,
    TokenKind.RELIC_SHOP,
    TokenKind.RELIC_EVENT_ROLL,
    TokenKind.RELIC_OWNED,
)

# Pack order within the Potion Entity kind; every source is engine-bounded
TOKEN_KINDS_POTION = (
    TokenKind.POTION_OWNED,
    TokenKind.POTION_REWARD,
    TokenKind.POTION_SHOP,
    TokenKind.POTION_EVENT_ROLL,
)

# The Entity kinds in window order, each its token kinds; the last four hold a single token kind
TOKEN_KINDS_OF_ENTITY_KIND = (
    TOKEN_KINDS_CARD,
    TOKEN_KINDS_RELIC,
    TOKEN_KINDS_POTION,
    (TokenKind.MONSTERS,),
    (TokenKind.EVENT_OPTIONS,),
    (TokenKind.CHARACTER,),
    (TokenKind.ROOMS,),
)

# Each token kind's token cap: the slice token traversal applies to that collection
MAP_TOKEN_KIND_MAX_TOKENS: dict[TokenKind, int] = {
    TokenKind.CARD_PILE_HAND: MAX_SIZE_CARD_PILE_HAND,
    TokenKind.CARD_PILE_DISCOVER: MAX_SIZE_CARD_PILE_DISCOVER,
    TokenKind.CARD_PILE_SELECTED: MAX_SIZE_CARD_PILE_SELECTED,
    TokenKind.CARD_REWARD_0: MAX_CARD_REWARD_ROLL,
    TokenKind.CARD_REWARD_1: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_2: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_3: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_REWARD_4: MAX_COMBAT_CARD_REWARD,
    TokenKind.CARD_SHOP: MAX_SIZE_SHOP_CARDS,
    TokenKind.CARD_PILE_EVENT_ROLL: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    TokenKind.CARD_MATCH_FLIPPED: 1,  # Match and Keep's face-up Card
    TokenKind.CARD_MATCH_PAIRS: MAX_SIZE_CARD_MATCH_PAIRS,
    TokenKind.CARD_PILE_STASIS: MAX_SIZE_CARD_PILE_STASIS,
    TokenKind.CARD_PLAY_QUEUE: MAX_SIZE_CARD_PLAY_QUEUE,
    TokenKind.CARD_PILE_NIGHTMARE: MAX_SIZE_CARD_PILE_NIGHTMARE,
    TokenKind.CARD_PILE_DRAW: MAX_SIZE_CARD_PILE_DRAW,
    TokenKind.CARD_PILE_DISCARD: MAX_SIZE_CARD_PILE_DISCARD,
    TokenKind.CARD_PILE_EXHAUST: MAX_SIZE_CARD_PILE_EXHAUST,
    TokenKind.CARD_PILE_DECK: MAX_SIZE_CARD_PILE_DECK,
    TokenKind.RELIC_REWARD: MAX_RELIC_REWARDS,
    TokenKind.RELIC_SHOP: MAX_SHOP_RELICS,
    TokenKind.RELIC_EVENT_ROLL: MAX_SIZE_RELIC_EVENT_ROLL,
    TokenKind.RELIC_OWNED: MAX_RELICS,
    TokenKind.POTION_OWNED: MAX_POTION_SLOTS,
    TokenKind.POTION_REWARD: MAX_POTION_REWARDS,
    TokenKind.POTION_SHOP: MAX_SHOP_POTIONS,
    TokenKind.POTION_EVENT_ROLL: MAX_SIZE_POTION_EVENT_ROLL,
    TokenKind.MONSTERS: MAX_MONSTERS,
    TokenKind.EVENT_OPTIONS: MAX_EVENT_OPTIONS,
    TokenKind.CHARACTER: 1,
    TokenKind.ROOMS: MAP_WIDTH,
}

# The cap of the block every collection packs into: the sum of the collections' caps
MAX_TOKENS = sum(MAP_TOKEN_KIND_MAX_TOKENS.values())


def _check_tables() -> None:
    """Check that the tables that restate the engine or an enum still agree with it."""
    if len(TOKEN_KINDS_CARD_REWARDS) != MAX_REWARD_CARD_BUNDLES:
        raise RuntimeError(
            f"The engine stages up to {MAX_REWARD_CARD_BUNDLES} reward bundles: "
            "add CARD_REWARD_<b> token kinds"
        )
    if set(MAP_TOKEN_KIND_MAX_TOKENS) != set(TokenKind):
        raise RuntimeError(
            "A TokenKind has no entry in MAP_TOKEN_KIND_MAX_TOKENS: "
            f"{set(TokenKind) - set(MAP_TOKEN_KIND_MAX_TOKENS)}"
        )

    # TokenKind is declared in packing order, valued 0..len-1: the Entity kind tables, one after
    # another, list it, every token kind in one table
    token_kinds = [
        token_kind
        for token_kinds_entity_kind in TOKEN_KINDS_OF_ENTITY_KIND
        for token_kind in token_kinds_entity_kind
    ]
    if token_kinds != list(TokenKind) or token_kinds != list(range(len(token_kinds))):
        names = [token_kind.name for token_kind in token_kinds]
        raise RuntimeError(
            "Declare TokenKind in packing order, valued 0..len-1: TOKEN_KINDS_OF_ENTITY_KIND's "
            f"entity kinds, one after another, list {names}"
        )


_check_tables()
