import math

import numpy as np
from slai import Effect
from slai import EffectDebuffsClear
from slai import GameEnv
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
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.encoding.shared import encode_shared_health_block_into
from src.rl.encoding.shared import encode_shared_stasis_slot_into


_IDX_OF_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
_IDX_OF_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
_INTENT_BLOCK_KINDS = {
    IntentKind.Block,
    IntentKind.AttackBlock,
    IntentKind.BlockBuff,
    IntentKind.BlockDebuff,
}
_INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
_INTENT_DEBUFF_KINDS = {
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


_CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
_OFF_MONSTER_NAME = _alloc("name", len(_IDX_OF_MONSTER_NAME))
_OFF_MONSTER_INTENT_DAMAGE = _alloc("intent_damage", _DAMAGE_DIM)
_OFF_MONSTER_INTENT_DAMAGE_SCALAR = _alloc("intent_damage_scalar")
_OFF_MONSTER_INTENT_INSTANCES = _alloc("intent_instances")
_OFF_MONSTER_INTENT_DAMAGE_TOTAL = _alloc("intent_damage_total", _DAMAGE_TOTAL_DIM)
_OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR = _alloc("intent_damage_total_scalar")
_OFF_MONSTER_INTENT_HAS_BLOCK = _alloc("intent_has_block")
_OFF_MONSTER_INTENT_HAS_BUFF = _alloc("intent_has_buff")
_OFF_MONSTER_INTENT_HAS_DEBUFF = _alloc("intent_has_debuff")
_OFF_MONSTER_HIT_NO_HEALTH_LOST = _alloc("hit_no_health_lost")
_OFF_MONSTER_HIT_IS_LETHAL = _alloc("hit_is_lethal")
_OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc("max_health_magnitude")
_OFF_MONSTER_HEALTH_FRACTION = _alloc("health_fraction")
_OFF_MONSTER_POISON_TICKS_TO_KILL = _alloc("poison_ticks_to_kill", _POISON_TICKS_MAX)
_OFF_MONSTER_HEALTH_TO_TRIGGER = _alloc("health_to_trigger", _HEALTH_TO_TRIGGER_DIM)
_OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR = _alloc("health_to_trigger_scalar")
_OFF_MONSTER_GOLD_STOLEN = _alloc("gold_stolen")
_OFF_MONSTER_INTENT_KIND = _alloc("intent_kind", len(_IDX_OF_INTENT_KIND))
_OFF_MONSTER_INTENT_HIDDEN = _alloc("intent_hidden")
_OFF_MONSTER_MOVE_CURRENT = _alloc("move_current", MONSTER_MOVESET_MAX)
_OFF_MONSTER_MOVE_PREV_1 = _alloc("move_prev_1", MONSTER_MOVESET_MAX)
_OFF_MONSTER_MOVE_PREV_2 = _alloc("move_prev_2", MONSTER_MOVESET_MAX)
_OFF_MONSTER_MOVE_PREV_COUNT = _alloc("move_prev_count", _MOVE_PREV_COUNT_DIM)
_OFF_MONSTER_REPLAY_TARGET = _alloc("replay_target")
_OFF_MONSTER_HALF_DEAD = _alloc("half_dead")
_OFF_MONSTER_ULTIMATE_USED = _alloc("ultimate_used")

ENCODING_DIM_MONSTER = sum(w for _, w in _CHANNEL_LAYOUT_MONSTER)

# Most nonzero columns a Monster row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_MONSTER = len(_CHANNEL_LAYOUT_MONSTER)


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
    monster: Monster,
    incoming: Incoming | None,
    ascension: int,
    np_own: np.ndarray,
    np_shared: np.ndarray,
) -> None:
    """Write a Monster's token: its own columns, then its health/block and its slot.

    Parameters
    ----------
    monster : Monster
        The Monster.
    incoming : Incoming | None
        What ending the turn now would cost (incoming.py); None when Runic Dome hides the intents.
    ascension : int
        The game's ascension, which sets Invincible's cap on the health a Poison tick takes.
    np_own : np.ndarray
        (ENCODING_DIM_MONSTER,) float32. The Monster's own range of its token row, written in
        place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_own[_OFF_MONSTER_NAME + _IDX_OF_MONSTER_NAME[monster.name]] = 1.0
    np_own[_OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    np_own[_OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    np_own[_OFF_MONSTER_REPLAY_TARGET] = float(monster.replay_target)
    np_own[_OFF_MONSTER_HALF_DEAD] = float(monster.half_dead)
    np_own[_OFF_MONSTER_ULTIMATE_USED] = float(monster.ultimate_used)

    # Poison's horizon
    stacks_of_modifier_kind = {modifier.kind: modifier.stacks for modifier in monster.modifiers}
    poison_ticks = _poison_ticks_to_kill(monster, stacks_of_modifier_kind, ascension)
    if poison_ticks is not None:
        np_own[_OFF_MONSTER_POISON_TICKS_TO_KILL + poison_ticks - 1] = 1.0

    # HP to its trigger
    health_to_trigger = _health_to_trigger(monster, stacks_of_modifier_kind)
    if health_to_trigger is not None:
        health_to_trigger_bucket = get_piecewise_bucket(
            health_to_trigger, 0, _HEALTH_MAX, _HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD
        )
        np_own[_OFF_MONSTER_HEALTH_TO_TRIGGER + health_to_trigger_bucket] = 1.0
        np_own[_OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR] = get_sqrt_norm(
            health_to_trigger, _HEALTH_MAX
        )

    # Stolen gold
    np_own[_OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, GOLD_STOLEN_MAX)

    # Intent
    intent = monster.intent
    if intent is None or incoming is None:
        np_own[_OFF_MONSTER_INTENT_HIDDEN] = 1.0
    else:
        damage = intent.damage or 0
        damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
        np_own[_OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

        total_damage = damage * (intent.instances or 1)
        np_own[_OFF_MONSTER_INTENT_DAMAGE_SCALAR] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
        np_own[_OFF_MONSTER_INTENT_INSTANCES] = (
            min(intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
        )
        total_damage_bucket = get_piecewise_bucket(
            total_damage, 0, _DAMAGE_TOTAL_MAX, _LINEAR_SQRT_THRESHOLD
        )
        np_own[_OFF_MONSTER_INTENT_DAMAGE_TOTAL + total_damage_bucket] = 1.0
        np_own[_OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR] = (
            min(total_damage, _DAMAGE_TOTAL_MAX) / _DAMAGE_TOTAL_MAX
        )
        np_own[_OFF_MONSTER_INTENT_HAS_BLOCK] = float(intent.kind in _INTENT_BLOCK_KINDS)
        np_own[_OFF_MONSTER_INTENT_HAS_BUFF] = float(intent.kind in _INTENT_BUFF_KINDS)
        np_own[_OFF_MONSTER_INTENT_HAS_DEBUFF] = float(intent.kind in _INTENT_DEBUFF_KINDS)

        # Its own move in the walk: block spends across the Monsters, slot by slot
        health_lost = incoming.health_lost_of_monster[monster.id]
        np_own[_OFF_MONSTER_HIT_NO_HEALTH_LOST] = float(
            incoming.damage_of_monster[monster.id] > 0 and health_lost == 0
        )
        np_own[_OFF_MONSTER_HIT_IS_LETHAL] = float(
            health_lost > 0 and health_lost >= incoming.health_at_monsters
        )

        # The intent category flags above miss Escape / Sleep / Stunned / Unknown
        np_own[_OFF_MONSTER_INTENT_KIND + _IDX_OF_INTENT_KIND[intent.kind]] = 1.0

    # Move identity: the current move + the two before it
    if monster.move_current is not None:
        np_own[_OFF_MONSTER_MOVE_CURRENT + monster.move_current] = 1.0

    # Under Runic Dome the history holds only the executed moves: no current
    # entry ends it, so the previous ones sit at [-1] and [-2]
    shift = 0 if intent is None else 1
    move_history = monster.move_history
    if len(move_history) >= 1 + shift:
        np_own[_OFF_MONSTER_MOVE_PREV_1 + move_history[-1 - shift]] = 1.0
    if len(move_history) >= 2 + shift:
        np_own[_OFF_MONSTER_MOVE_PREV_2 + move_history[-2 - shift]] = 1.0
    move_prev_count_bucket = get_piecewise_bucket(
        max(len(move_history) - shift, 0), 0, GameEnv.MAX_MOVE_HISTORY, _LINEAR_SQRT_THRESHOLD
    )
    np_own[_OFF_MONSTER_MOVE_PREV_COUNT + move_prev_count_bucket] = 1.0

    # Health & Block
    encode_shared_health_block_into(monster.health, monster.block, np_shared)

    # Stasis slot. Shared w/ Cards in the stasis pile
    encode_shared_stasis_slot_into(monster.slot, np_shared)


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
