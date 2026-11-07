"""Feature layout and static catalogs for monster rows."""

import math

from slai import Effect
from slai import IntentKind
from slai import MonsterName
from slai import get_monster_templates

from src.rl.constants import FACTOR_VULN
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import MONSTER_MOVESET_MAX
from src.rl.schema.layout import make_channel_allocator
from src.rl.schema.scalars import get_piecewise_dim


MAP_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
MAP_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
INTENT_BLOCK_KINDS = {
    IntentKind.Block,
    IntentKind.AttackBlock,
    IntentKind.BlockBuff,
    IntentKind.BlockDebuff,
}
INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
INTENT_DEBUFF_KINDS = {
    IntentKind.Debuff,
    IntentKind.AttackDebuff,
    IntentKind.DebuffPowerful,
    IntentKind.BlockDebuff,
}

# Normalization caps, catalog-anchored
DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # Widest hit, Vulnerable-scaled
DAMAGE_TOTAL_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)  # Widest turn, likewise
INSTANCES_MAX = INTENT_INSTANCES_MAX
HEALTH_MAX = HEALTH_MAX_ENTITY  # The Transient's 999

LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, DAMAGE_MAX, LINEAR_SQRT_THRESHOLD)
_DAMAGE_TOTAL_DIM = get_piecewise_dim(0, DAMAGE_TOTAL_MAX, LINEAR_SQRT_THRESHOLD)


# Health/block is not here: it is the combatants' health/block columns (health_block.py)


CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
OFF_MONSTER_NAME = _alloc("name", len(MAP_MONSTER_NAME))
OFF_MONSTER_INTENT_DAMAGE = _alloc("intent_damage", _DAMAGE_DIM)
OFF_MONSTER_INTENT_DAMAGE_SCALAR = _alloc("intent_damage_scalar")
OFF_MONSTER_INTENT_INSTANCES = _alloc("intent_instances")
OFF_MONSTER_INTENT_DAMAGE_TOTAL = _alloc(
    "intent_damage_total", _DAMAGE_TOTAL_DIM
)  # Damage x instances
OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR = _alloc("intent_damage_total_scalar")
OFF_MONSTER_INTENT_HAS_BLOCK = _alloc("intent_has_block")
OFF_MONSTER_INTENT_HAS_BUFF = _alloc("intent_has_buff")
OFF_MONSTER_INTENT_HAS_DEBUFF = _alloc("intent_has_debuff")
OFF_MONSTER_HIT_FULLY_BLOCKED = _alloc("hit_fully_blocked")
OFF_MONSTER_HIT_IS_LETHAL = _alloc("hit_is_lethal")
OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc("max_health_magnitude")
OFF_MONSTER_HEALTH_FRACTION = _alloc("health_fraction")
OFF_MONSTER_POISON_IS_LETHAL = _alloc(
    "poison_is_lethal"
)  # Its next turn start's Poison tick kills it
OFF_MONSTER_GOLD_STOLEN = _alloc("gold_stolen")
OFF_MONSTER_INTENT_KIND = _alloc("intent_kind", len(MAP_INTENT_KIND))
OFF_MONSTER_INTENT_HIDDEN = _alloc(
    "intent_hidden"
)  # Runic Dome: every intent channel and hit flag reads 0
OFF_MONSTER_MOVE_CURRENT = _alloc(
    "move_current", MONSTER_MOVESET_MAX
)  # Index into the spawn-rolled moveset
OFF_MONSTER_MOVE_PREV_1 = _alloc(
    "move_prev_1", MONSTER_MOVESET_MAX
)  # One move back (move_history's last entry is the current move)
OFF_MONSTER_MOVE_PREV_2 = _alloc("move_prev_2", MONSTER_MOVESET_MAX)
OFF_MONSTER_REPLAY_TARGET = _alloc("replay_target")  # A waiting replay aims at it
OFF_MONSTER_HALF_DEAD = _alloc(
    "half_dead"
)  # Health and block read 0, no Card can target it (Awakened One, Darkling)
OFF_MONSTER_ULTIMATE_USED = _alloc("ultimate_used")  # Its once-per-Combat move has executed

ENCODING_DIM_MONSTER = sum(w for _, w in CHANNEL_LAYOUT_MONSTER)

# Most nonzero columns a Monster row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_MONSTER = len(CHANNEL_LAYOUT_MONSTER)


# Whole-moveset Effects per (ascension, MonsterName), collected once (one entry
# per distinct move)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, tuple[Effect, ...]]] = {}


def moveset_effects(ascension: int) -> dict[MonsterName, tuple[Effect, ...]]:
    """Return each Monster's catalog Effects at the requested ascension.

    Repeated moves contribute their Effects once. The catalog is collected once per ascension.

    Parameters
    ----------
    ascension : int
        The ascension whose Monster templates supply the movesets.

    Returns
    -------
    dict[MonsterName, tuple[Effect, ...]]
        Each Monster name's Effects in moveset order, shared with the catalog cache.
    """
    by_name = _MOVESET_EFFECTS.get(ascension)
    if by_name is None:
        by_name = {
            template.name: tuple(
                effect for move in dict.fromkeys(template.moves) for effect in move.effects
            )
            for template in get_monster_templates(ascension)
        }
        _MOVESET_EFFECTS[ascension] = by_name

    return by_name
