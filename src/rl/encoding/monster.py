import math

import numpy as np
from slai import Effect
from slai import GameState
from slai import IntentKind
from slai import ModifierKind
from slai import Monster
from slai import MonsterName
from slai import get_monster_templates

from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STOLEN_MAX
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import INVINCIBLE_STACKS_OF_ASCENSION
from src.rl.constants import MONSTER_MOVESET_MAX
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.types import make_channel_allocator


_MAP_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
_MAP_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
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
_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # Widest hit, Vulnerable-scaled
_DAMAGE_TOTAL_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)  # Widest turn, likewise
_INSTANCES_MAX = INTENT_INSTANCES_MAX
_HEALTH_MAX = HEALTH_MAX_ENTITY  # The Transient's 999

_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
_DAMAGE_TOTAL_DIM = get_piecewise_dim(0, _DAMAGE_TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)


# Health/block is not here: it is the combatants' health/block columns (health_block.py)


CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
OFF_MONSTER_NAME = _alloc("name", len(_MAP_MONSTER_NAME))
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
OFF_MONSTER_INTENT_KIND = _alloc("intent_kind", len(_MAP_INTENT_KIND))
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


def _intent_total_damage(monster: Monster) -> int:
    intent = monster.intent
    if intent is None:  # Runic Dome hides it
        return 0
    return (intent.damage or 0) * (intent.instances or 1)


def encode_monster_into(
    monster: Monster,
    character_health: int,
    character_block: int,
    get_outgoing_damage: int | None,
    ascension: int,
    out: np.ndarray,
) -> None:
    out[OFF_MONSTER_NAME + _MAP_MONSTER_NAME[monster.name]] = 1.0
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    out[OFF_MONSTER_REPLAY_TARGET] = float(monster.replay_target)
    out[OFF_MONSTER_HALF_DEAD] = float(monster.half_dead)
    out[OFF_MONSTER_ULTIMATE_USED] = float(monster.ultimate_used)

    # Poison-is-lethal
    stacks_of_modifier_kind = {modifier.kind: modifier.stacks for modifier in monster.modifiers}
    poison_health_loss = stacks_of_modifier_kind.get(ModifierKind.Poison, 0)
    if ModifierKind.Buffer in stacks_of_modifier_kind:
        poison_health_loss = 0
    elif ModifierKind.Intangible in stacks_of_modifier_kind:
        poison_health_loss = min(poison_health_loss, 1)
    elif ModifierKind.Invincible in stacks_of_modifier_kind:
        poison_health_loss = min(poison_health_loss, INVINCIBLE_STACKS_OF_ASCENSION[ascension])
    out[OFF_MONSTER_POISON_IS_LETHAL] = float(
        poison_health_loss > 0 and poison_health_loss >= monster.health
    )

    # Stolen gold
    out[OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, GOLD_STOLEN_MAX)

    # Intent
    intent = monster.intent
    if intent is None:
        out[OFF_MONSTER_INTENT_HIDDEN] = 1.0
    else:
        damage = intent.damage or 0
        damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
        out[OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

        total_damage = _intent_total_damage(monster)
        out[OFF_MONSTER_INTENT_DAMAGE_SCALAR] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
        out[OFF_MONSTER_INTENT_INSTANCES] = (
            min(intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
        )
        total_damage_bucket = get_piecewise_bucket(
            total_damage, 0, _DAMAGE_TOTAL_MAX, _LINEAR_SQRT_THRESHOLD
        )
        out[OFF_MONSTER_INTENT_DAMAGE_TOTAL + total_damage_bucket] = 1.0
        out[OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR] = (
            min(total_damage, _DAMAGE_TOTAL_MAX) / _DAMAGE_TOTAL_MAX
        )
        out[OFF_MONSTER_INTENT_HAS_BLOCK] = float(intent.kind in INTENT_BLOCK_KINDS)
        out[OFF_MONSTER_INTENT_HAS_BUFF] = float(intent.kind in INTENT_BUFF_KINDS)
        out[OFF_MONSTER_INTENT_HAS_DEBUFF] = float(intent.kind in INTENT_DEBUFF_KINDS)
        out[OFF_MONSTER_HIT_FULLY_BLOCKED] = float(total_damage <= character_block)
        out[OFF_MONSTER_HIT_IS_LETHAL] = float(
            get_outgoing_damage >= character_health + character_block
        )

        # The intent category flags above miss Escape / Sleep / Stunned / Unknown
        out[OFF_MONSTER_INTENT_KIND + _MAP_INTENT_KIND[intent.kind]] = 1.0

    # Move identity: the current move + the two before it
    if monster.move_current is not None:
        out[OFF_MONSTER_MOVE_CURRENT + monster.move_current] = 1.0

    # Under Runic Dome the history holds only the executed moves: no current
    # entry ends it, so the previous ones sit at [-1] and [-2]
    shift = 0 if intent is None else 1
    move_history = monster.move_history
    if len(move_history) >= 1 + shift:
        out[OFF_MONSTER_MOVE_PREV_1 + move_history[-1 - shift]] = 1.0
    if len(move_history) >= 2 + shift:
        out[OFF_MONSTER_MOVE_PREV_2 + move_history[-2 - shift]] = 1.0


def get_outgoing_damage(monsters: list[Monster]) -> int | None:
    # Runic Dome hides every intent, so the incoming damage is unknown
    if any(monster.intent is None for monster in monsters):
        return None

    return sum(_intent_total_damage(monster) for monster in monsters)


# Whole-moveset Effects per (ascension, MonsterName), collected once (one entry
# per distinct move)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, tuple[Effect, ...]]] = {}


def moveset_effects(ascension: int) -> dict[MonsterName, tuple[Effect, ...]]:
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


def get_monster_entities(game_state: GameState) -> list[Monster]:
    """Return the Monsters behind the Monster tokens, in slot order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Monster]
        The Combat's Monsters; empty outside Combat.
    """
    combat = game_state.combat
    return combat.monsters if combat is not None else []
