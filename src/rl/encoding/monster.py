import math
from enum import StrEnum
from enum import auto

import numpy as np
from slai import Effect
from slai import EffectDebuffsClear
from slai import GameEnv
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
from src.rl.constants import MONSTER_MOVESET_MAX
from src.rl.encoding.incoming import Incoming
from src.rl.encoding.incoming import get_poison_tick
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

# Its turn starts until Poison alone kills it: past this, the channel reads none
_POISON_TICKS_MAX = 15

# Champ's Anger and Time Eater's Haste come below half HP, once per Combat
_MONSTER_NAMES_HALF_HEALTH_ULTIMATE = {MonsterName.Champ, MonsterName.TimeEater}

# Exact through a turn's burst
_HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD = 49
_HEALTH_TO_TRIGGER_DIM = get_piecewise_dim(
    0, _HEALTH_MAX, _HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD
)

_MOVE_PREV_COUNT_DIM = get_piecewise_dim(0, GameEnv.MAX_MOVE_HISTORY, _LINEAR_SQRT_THRESHOLD)


# Health/block is not here: it is the combatants' health/block columns (health_block.py)
class MonsterChannel(StrEnum):
    NAME = auto()
    INTENT_DAMAGE = auto()
    INTENT_DAMAGE_SCALAR = auto()
    INTENT_INSTANCES = auto()
    INTENT_DAMAGE_TOTAL = auto()  # Damage x instances
    INTENT_DAMAGE_TOTAL_SCALAR = auto()
    INTENT_HAS_BLOCK = auto()
    INTENT_HAS_BUFF = auto()
    INTENT_HAS_DEBUFF = auto()
    HIT_NO_HEALTH_LOST = auto()  # It deals damage in the walk (incoming.py), and that costs no HP
    HIT_IS_LETHAL = auto()  # What it costs in the walk kills from the HP the Monsters start at
    MAX_HEALTH_MAGNITUDE = auto()
    HEALTH_FRACTION = auto()
    POISON_TICKS_TO_KILL = auto()  # Its turn starts until Poison alone kills it, bucket 0 the next
    HEALTH_TO_TRIGGER = auto()  # The HP loss that fires Split, Mode Shift, Anger or Haste
    HEALTH_TO_TRIGGER_SCALAR = auto()
    GOLD_STOLEN = auto()
    INTENT_KIND = auto()
    INTENT_HIDDEN = auto()  # Runic Dome: every intent channel and hit flag reads 0
    MOVE_CURRENT = auto()  # Index into the spawn-rolled moveset
    MOVE_PREV_1 = auto()  # One move back (move_history's last entry is the current move)
    MOVE_PREV_2 = auto()
    MOVE_PREV_COUNT = auto()  # How many moves lie behind the current one
    REPLAY_TARGET = auto()  # A waiting replay aims at it
    HALF_DEAD = auto()  # Health and block read 0, no Card can target it (Awakened One, Darkling)
    ULTIMATE_USED = auto()  # Its once-per-Combat move has executed


CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
OFF_MONSTER_NAME = _alloc(MonsterChannel.NAME, len(_MAP_MONSTER_NAME))
OFF_MONSTER_INTENT_DAMAGE = _alloc(MonsterChannel.INTENT_DAMAGE, _DAMAGE_DIM)
OFF_MONSTER_INTENT_DAMAGE_SCALAR = _alloc(MonsterChannel.INTENT_DAMAGE_SCALAR)
OFF_MONSTER_INTENT_INSTANCES = _alloc(MonsterChannel.INTENT_INSTANCES)
OFF_MONSTER_INTENT_DAMAGE_TOTAL = _alloc(MonsterChannel.INTENT_DAMAGE_TOTAL, _DAMAGE_TOTAL_DIM)
OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR = _alloc(MonsterChannel.INTENT_DAMAGE_TOTAL_SCALAR)
OFF_MONSTER_INTENT_HAS_BLOCK = _alloc(MonsterChannel.INTENT_HAS_BLOCK)
OFF_MONSTER_INTENT_HAS_BUFF = _alloc(MonsterChannel.INTENT_HAS_BUFF)
OFF_MONSTER_INTENT_HAS_DEBUFF = _alloc(MonsterChannel.INTENT_HAS_DEBUFF)
OFF_MONSTER_HIT_NO_HEALTH_LOST = _alloc(MonsterChannel.HIT_NO_HEALTH_LOST)
OFF_MONSTER_HIT_IS_LETHAL = _alloc(MonsterChannel.HIT_IS_LETHAL)
OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc(MonsterChannel.MAX_HEALTH_MAGNITUDE)
OFF_MONSTER_HEALTH_FRACTION = _alloc(MonsterChannel.HEALTH_FRACTION)
OFF_MONSTER_POISON_TICKS_TO_KILL = _alloc(MonsterChannel.POISON_TICKS_TO_KILL, _POISON_TICKS_MAX)
OFF_MONSTER_HEALTH_TO_TRIGGER = _alloc(MonsterChannel.HEALTH_TO_TRIGGER, _HEALTH_TO_TRIGGER_DIM)
OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR = _alloc(MonsterChannel.HEALTH_TO_TRIGGER_SCALAR)
OFF_MONSTER_GOLD_STOLEN = _alloc(MonsterChannel.GOLD_STOLEN)
OFF_MONSTER_INTENT_KIND = _alloc(MonsterChannel.INTENT_KIND, len(_MAP_INTENT_KIND))
OFF_MONSTER_INTENT_HIDDEN = _alloc(MonsterChannel.INTENT_HIDDEN)
OFF_MONSTER_MOVE_CURRENT = _alloc(MonsterChannel.MOVE_CURRENT, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_1 = _alloc(MonsterChannel.MOVE_PREV_1, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_2 = _alloc(MonsterChannel.MOVE_PREV_2, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_COUNT = _alloc(MonsterChannel.MOVE_PREV_COUNT, _MOVE_PREV_COUNT_DIM)
OFF_MONSTER_REPLAY_TARGET = _alloc(MonsterChannel.REPLAY_TARGET)
OFF_MONSTER_HALF_DEAD = _alloc(MonsterChannel.HALF_DEAD)
OFF_MONSTER_ULTIMATE_USED = _alloc(MonsterChannel.ULTIMATE_USED)

ENCODING_DIM_MONSTER = sum(w for _, w in CHANNEL_LAYOUT_MONSTER)

# Most nonzero columns a Monster row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_MONSTER = len(CHANNEL_LAYOUT_MONSTER)


def _poison_ticks_to_kill(
    monster: Monster, stacks_of_modifier_kind: dict[ModifierKind, int], ascension: int
) -> int | None:
    """Count the Monster's turn starts until its Poison alone kills it.

    Each tick takes the stacks and then drops them by 1, Regeneration heals between ticks, and
    Nemesis holds Intangible at every other tick.

    Parameters
    ----------
    monster : Monster
        The Monster.
    stacks_of_modifier_kind : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.
    ascension : int
        The run's Ascension.

    Returns
    -------
    int | None
        The ticks, 1 the next one; None when Poison wears off first, the kill is past
        _POISON_TICKS_MAX, a move ends it first (a Split's children hold no Poison, Anger and Haste
        clear it), or the Monster is half-dead, which no HP loss reaches.
    """
    if monster.half_dead:
        return None

    poison = stacks_of_modifier_kind.get(ModifierKind.Poison, 0)
    regeneration = stacks_of_modifier_kind.get(ModifierKind.Regeneration, 0)
    intangible = ModifierKind.Intangible in stacks_of_modifier_kind
    can_roll_debuffs_clear = (
        monster.name in _MONSTER_NAMES_HALF_HEALTH_ULTIMATE and not monster.ultimate_used
    )

    # Anger or Haste clears the Poison on its move, behind that turn's tick
    debuffs_clear_next = can_roll_debuffs_clear and any(
        isinstance(effect, EffectDebuffsClear) for effect in monster.move_effects
    )
    health = monster.health
    for ticks in range(1, min(poison, _POISON_TICKS_MAX) + 1):
        health -= get_poison_tick(
            poison - ticks + 1, intangible, stacks_of_modifier_kind, ascension
        )
        if health <= 0:
            return ticks
        if debuffs_clear_next or (
            ModifierKind.Splittable in stacks_of_modifier_kind
            and health <= monster.health_max // 2
        ):
            return None

        # Below half, the roll behind its move is Anger or Haste; every Nemesis move regains
        # Intangible unless it still holds it
        debuffs_clear_next = can_roll_debuffs_clear and health < monster.health_max // 2
        intangible = monster.name == MonsterName.Nemesis and not intangible
        health = min(health + regeneration, monster.health_max)

    return None


def _health_to_trigger(
    monster: Monster, stacks_of_modifier_kind: dict[ModifierKind, int]
) -> int | None:
    """Return the HP loss that fires the Monster's HP trigger.

    Parameters
    ----------
    monster : Monster
        The Monster.
    stacks_of_modifier_kind : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.

    Returns
    -------
    int | None
        The HP, 0 when the trigger has fired; None for a Monster with no trigger waiting.
    """
    if ModifierKind.ModeShift in stacks_of_modifier_kind:
        return max(stacks_of_modifier_kind[ModifierKind.ModeShift], 0)

    health_half = monster.health_max // 2

    # Split at half or less, the once-per-Combat moves below half
    if ModifierKind.Splittable in stacks_of_modifier_kind:
        return max(monster.health - health_half, 0)
    if monster.name in _MONSTER_NAMES_HALF_HEALTH_ULTIMATE and not monster.ultimate_used:
        return max(monster.health - health_half + 1, 0)
    return None


def encode_monster_into(
    monster: Monster, incoming: Incoming | None, ascension: int, out: np.ndarray
) -> None:
    out[OFF_MONSTER_NAME + _MAP_MONSTER_NAME[monster.name]] = 1.0
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    out[OFF_MONSTER_REPLAY_TARGET] = float(monster.replay_target)
    out[OFF_MONSTER_HALF_DEAD] = float(monster.half_dead)
    out[OFF_MONSTER_ULTIMATE_USED] = float(monster.ultimate_used)

    # Poison's horizon
    stacks_of_modifier_kind = {modifier.kind: modifier.stacks for modifier in monster.modifiers}
    poison_ticks = _poison_ticks_to_kill(monster, stacks_of_modifier_kind, ascension)
    if poison_ticks is not None:
        out[OFF_MONSTER_POISON_TICKS_TO_KILL + poison_ticks - 1] = 1.0

    # HP to its trigger
    health_to_trigger = _health_to_trigger(monster, stacks_of_modifier_kind)
    if health_to_trigger is not None:
        health_to_trigger_bucket = get_piecewise_bucket(
            health_to_trigger, 0, _HEALTH_MAX, _HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD
        )
        out[OFF_MONSTER_HEALTH_TO_TRIGGER + health_to_trigger_bucket] = 1.0
        out[OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR] = get_sqrt_norm(health_to_trigger, _HEALTH_MAX)

    # Stolen gold
    out[OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, GOLD_STOLEN_MAX)

    # Intent
    intent = monster.intent
    if intent is None or incoming is None:
        out[OFF_MONSTER_INTENT_HIDDEN] = 1.0
    else:
        damage = intent.damage or 0
        damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
        out[OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

        total_damage = damage * (intent.instances or 1)
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

        # Its own move in the walk: block spends across the Monsters, slot by slot
        health_lost = incoming.health_lost_of_monster[monster.id]
        out[OFF_MONSTER_HIT_NO_HEALTH_LOST] = float(
            incoming.damage_of_monster[monster.id] > 0 and health_lost == 0
        )
        out[OFF_MONSTER_HIT_IS_LETHAL] = float(
            health_lost > 0 and health_lost >= incoming.health_at_monsters
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
    move_prev_count_bucket = get_piecewise_bucket(
        max(len(move_history) - shift, 0), 0, GameEnv.MAX_MOVE_HISTORY, _LINEAR_SQRT_THRESHOLD
    )
    out[OFF_MONSTER_MOVE_PREV_COUNT + move_prev_count_bucket] = 1.0


# Whole-moveset Effects per (ascension, MonsterName), collected once (one entry
# per distinct move)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, tuple[Effect, ...]]] = {}


def moveset_effects(ascension: int) -> dict[MonsterName, tuple[Effect, ...]]:
    by_name = _MOVESET_EFFECTS.get(ascension)
    if by_name is None:
        by_name = {
            template.name: tuple(
                effect
                for move in dict.fromkeys(template.moves)
                for effect in move.effects
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
