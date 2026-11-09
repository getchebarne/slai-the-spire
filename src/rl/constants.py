"""Constants for the RL module.

Centralizes magic numbers and configuration values used across the RL codebase.
"""

import slai as _slai

from src.rl.types import CollectionKind
from src.rl.types import EntityKind


# The policy's action types are exactly the engine's
ACTION_TYPES = _slai.ActionType.members()
NUM_ACTION_TYPES = len(ACTION_TYPES)

# The top Ascension, which GameEnv doesn't expose; it sizes the ascension thermometer
# (a checkpoint shape) and every catalog fold
ASCENSION_LEVEL = 20

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
    max(
        move.damage * move.instances
        for move in _MONSTER_MOVES
        if move.damage is not None and move.instances is not None
    ),
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
# CARD_REWARD_<b> collection kind, so a Reward token's collection kind names its bundle
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
MAX_RELICS = 35

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

# Batches repeat Maps and tokens from here up: above an Actor step's batch, below any minibatch
MIN_BATCH_DEDUP = 1024


COLLECTION_KINDS_CARD_REWARDS = (  # One collection kind per Reward bundle, in bundle order
    CollectionKind.CARD_REWARD_0,
    CollectionKind.CARD_REWARD_1,
    CollectionKind.CARD_REWARD_2,
    CollectionKind.CARD_REWARD_3,
    CollectionKind.CARD_REWARD_4,
)

# The Card collection kinds in window order: decision pools first, observational piles last
COLLECTION_KINDS_CARD = (
    CollectionKind.CARD_PILE_HAND,
    CollectionKind.CARD_PILE_DISCOVER,
    CollectionKind.CARD_PILE_SELECTED,
    *COLLECTION_KINDS_CARD_REWARDS,
    CollectionKind.CARD_SHOP,
    CollectionKind.CARD_PILE_EVENT_ROLL,
    CollectionKind.CARD_MATCH_FLIPPED,
    CollectionKind.CARD_MATCH_PAIRS,
    CollectionKind.CARD_PILE_STASIS,
    CollectionKind.CARD_PLAY_QUEUE,
    CollectionKind.CARD_PILE_NIGHTMARE,
    CollectionKind.CARD_PILE_DRAW,
    CollectionKind.CARD_PILE_DISCARD,
    CollectionKind.CARD_PILE_EXHAUST,
    CollectionKind.CARD_PILE_DECK,
)

# The Relic collection kinds in window order: decision pools first, the owned Relics last
_COLLECTION_KINDS_RELIC = (
    CollectionKind.RELIC_REWARD,
    CollectionKind.RELIC_SHOP,
    CollectionKind.RELIC_EVENT_ROLL,
    CollectionKind.RELIC_OWNED,
)

# The Potion collection kinds in window order; every source is engine-bounded
_COLLECTION_KINDS_POTION = (
    CollectionKind.POTION_OWNED,
    CollectionKind.POTION_REWARD,
    CollectionKind.POTION_SHOP,
    CollectionKind.POTION_EVENT_ROLL,
)

# The Entity kinds in window order, each its collection kinds; the last four hold a single
# collection kind
COLLECTION_KINDS_OF_ENTITY_KIND: dict[EntityKind, tuple[CollectionKind, ...]] = {
    EntityKind.CARD: COLLECTION_KINDS_CARD,
    EntityKind.RELIC: _COLLECTION_KINDS_RELIC,
    EntityKind.POTION: _COLLECTION_KINDS_POTION,
    EntityKind.MONSTER: (CollectionKind.MONSTERS,),
    EntityKind.EVENT_OPTION: (CollectionKind.EVENT_OPTIONS,),
    EntityKind.CHARACTER: (CollectionKind.CHARACTER,),
    EntityKind.ROOM: (CollectionKind.ROOMS,),
}

# A state's tokens come in this order: the Entity kinds one after another, each its
# collection kinds
COLLECTION_KINDS_IN_WINDOW_ORDER = tuple(
    collection_kind
    for collection_kinds_entity_kind in COLLECTION_KINDS_OF_ENTITY_KIND.values()
    for collection_kind in collection_kinds_entity_kind
)

# Each collection kind's token cap: the most tokens of that collection kind a state holds
MAX_TOKENS_OF_COLLECTION_KIND: dict[CollectionKind, int] = {
    CollectionKind.CARD_PILE_HAND: MAX_SIZE_CARD_PILE_HAND,
    CollectionKind.CARD_PILE_DISCOVER: MAX_SIZE_CARD_PILE_DISCOVER,
    CollectionKind.CARD_PILE_SELECTED: MAX_SIZE_CARD_PILE_SELECTED,
    CollectionKind.CARD_REWARD_0: MAX_CARD_REWARD_ROLL,
    CollectionKind.CARD_REWARD_1: MAX_COMBAT_CARD_REWARD,
    CollectionKind.CARD_REWARD_2: MAX_COMBAT_CARD_REWARD,
    CollectionKind.CARD_REWARD_3: MAX_COMBAT_CARD_REWARD,
    CollectionKind.CARD_REWARD_4: MAX_COMBAT_CARD_REWARD,
    CollectionKind.CARD_SHOP: MAX_SIZE_SHOP_CARDS,
    CollectionKind.CARD_PILE_EVENT_ROLL: MAX_SIZE_CARD_PILE_EVENT_ROLL,
    CollectionKind.CARD_MATCH_FLIPPED: 1,
    CollectionKind.CARD_MATCH_PAIRS: MAX_SIZE_CARD_MATCH_PAIRS,
    CollectionKind.CARD_PILE_STASIS: MAX_SIZE_CARD_PILE_STASIS,
    CollectionKind.CARD_PLAY_QUEUE: MAX_SIZE_CARD_PLAY_QUEUE,
    CollectionKind.CARD_PILE_NIGHTMARE: MAX_SIZE_CARD_PILE_NIGHTMARE,
    CollectionKind.CARD_PILE_DRAW: MAX_SIZE_CARD_PILE_DRAW,
    CollectionKind.CARD_PILE_DISCARD: MAX_SIZE_CARD_PILE_DISCARD,
    CollectionKind.CARD_PILE_EXHAUST: MAX_SIZE_CARD_PILE_EXHAUST,
    CollectionKind.CARD_PILE_DECK: MAX_SIZE_CARD_PILE_DECK,
    CollectionKind.RELIC_REWARD: MAX_RELIC_REWARDS,
    CollectionKind.RELIC_SHOP: MAX_SHOP_RELICS,
    CollectionKind.RELIC_EVENT_ROLL: MAX_SIZE_RELIC_EVENT_ROLL,
    CollectionKind.RELIC_OWNED: MAX_RELICS,
    CollectionKind.POTION_OWNED: MAX_POTION_SLOTS,
    CollectionKind.POTION_REWARD: MAX_POTION_REWARDS,
    CollectionKind.POTION_SHOP: MAX_SHOP_POTIONS,
    CollectionKind.POTION_EVENT_ROLL: MAX_SIZE_POTION_EVENT_ROLL,
    CollectionKind.MONSTERS: MAX_MONSTERS,
    CollectionKind.EVENT_OPTIONS: MAX_EVENT_OPTIONS,
    CollectionKind.CHARACTER: 1,
    CollectionKind.ROOMS: MAP_WIDTH,
}

# The cap of the block every collection packs into: the sum of the collections' caps
MAX_TOKENS = sum(MAX_TOKENS_OF_COLLECTION_KIND.values())


def _check_tables() -> None:
    """Check that the tables that restate the engine or an enum still agree with it."""
    if len(COLLECTION_KINDS_CARD_REWARDS) != MAX_REWARD_CARD_BUNDLES:
        raise RuntimeError(
            f"The engine stages up to {MAX_REWARD_CARD_BUNDLES} reward bundles: "
            "add CARD_REWARD_<b> collection kinds"
        )
    if set(MAX_TOKENS_OF_COLLECTION_KIND) != set(CollectionKind):
        raise RuntimeError(
            "A CollectionKind has no entry in MAX_TOKENS_OF_COLLECTION_KIND: "
            f"{set(CollectionKind) - set(MAX_TOKENS_OF_COLLECTION_KIND)}"
        )
    if set(COLLECTION_KINDS_OF_ENTITY_KIND) != set(EntityKind):
        raise RuntimeError(
            "An EntityKind has no entry in COLLECTION_KINDS_OF_ENTITY_KIND: "
            f"{set(EntityKind) - set(COLLECTION_KINDS_OF_ENTITY_KIND)}"
        )

    # Every collection kind sits once in the window order, and CollectionKind is valued 0..len-1
    collection_kinds = sorted(COLLECTION_KINDS_IN_WINDOW_ORDER)
    if (
        collection_kinds != list(CollectionKind)
        or collection_kinds != list(range(len(collection_kinds)))
    ):
        names = [collection_kind.name for collection_kind in COLLECTION_KINDS_IN_WINDOW_ORDER]
        raise RuntimeError(
            "Each CollectionKind must sit once in COLLECTION_KINDS_OF_ENTITY_KIND's tables, and "
            f"CollectionKind must be valued 0..len-1; the tables list {names}"
        )


_check_tables()
