import math
from collections.abc import Iterable

import numpy as np
from slai import FACTOR_VULN
from slai import MAX_MOVE_HISTORY
from slai import Effect
from slai import EffectDebuffsClear
from slai import IntentKind
from slai import ModifierKind
from slai import Monster
from slai import MonsterMoveTemplate
from slai import MonsterName
from slai import get_monster_templates

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import MONSTER_TEMPLATES
from src.rl.constants import SCALE_GOLD
from src.rl.constants import SCALE_HEALTH
from src.rl.encoding.damage import get_poison_tick
from src.rl.encoding.damage import get_stacks_of_modifier_kind
from src.rl.encoding.damage import make_combatant
from src.rl.encoding.incoming import Incoming
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import encode_piecewise_into
from src.rl.encoding.scalars import get_linear_norm
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.shared import encode_shared_health_block_into
from src.rl.encoding.shared import encode_shared_outgoing_into
from src.rl.encoding.shared import encode_shared_stasis_slot_into
from src.rl.types import EntityKind


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

# Caps, catalog-anchored: the two-hots' maxima and the instances' scale
_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # Widest hit, Vulnerable-scaled
_DAMAGE_TOTAL_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)  # Widest turn, likewise
_INSTANCES_MAX = INTENT_INSTANCES_MAX
_HEALTH_MAX = HEALTH_MAX_ENTITY  # The Transient's 999

_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
_DAMAGE_TOTAL_DIM = get_piecewise_dim(0, _DAMAGE_TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)

# Max HP: the shared block's health knots, from 1
_MAX_HEALTH_LINEAR_SQRT_THRESHOLD = 25
_DIM_MAX_HEALTH_OHE = get_piecewise_dim(1, _HEALTH_MAX, _MAX_HEALTH_LINEAR_SQRT_THRESHOLD)

# Its turn starts until Poison alone kills it: past this, the channel reads none
_POISON_TICKS_MAX = 10

# Champ's Anger and Time Eater's Haste come below half HP, once per Combat
_MONSTER_NAMES_HALF_HEALTH_ULTIMATE = {MonsterName.Champ, MonsterName.TimeEater}


def _health_to_trigger(
    name: MonsterName,
    health: int,
    health_max: int,
    ultimate_used: bool,
    stacks_of_modifier_kind: dict[ModifierKind, int],
) -> int | None:
    """Return the HP loss that fires the Monster's HP trigger.

    Parameters
    ----------
    name : MonsterName
        The Monster's name.
    health : int
        The Monster's health.
    health_max : int
        The Monster's max health.
    ultimate_used : bool
        Whether the Monster has used its once-per-Combat move below half HP.
    stacks_of_modifier_kind : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.

    Returns
    -------
    int | None
        The HP, 0 when the trigger has fired; None for a Monster with no trigger waiting.
    """
    if ModifierKind.ModeShift in stacks_of_modifier_kind:
        return max(stacks_of_modifier_kind[ModifierKind.ModeShift], 0)

    health_half = health_max // 2

    # Split at half or less, the once-per-Combat moves below half
    if ModifierKind.Splittable in stacks_of_modifier_kind:
        return max(health - health_half, 0)

    if name in _MONSTER_NAMES_HALF_HEALTH_ULTIMATE and not ultimate_used:
        return max(health - health_half + 1, 0)

    return None


# The trigger's largest HP loss: each template at its top HP roll, its Modifiers as spawned
_HEALTH_TO_TRIGGER_MAX = max(
    _health_to_trigger(
        template.name,
        template.hp_range[1],
        template.hp_range[1],
        False,
        {spawn.kind: spawn.stacks_max for spawn in template.modifier_spawns},
    )
    or 0
    for template in MONSTER_TEMPLATES
)

# Exact through a turn's burst
_HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD = 49
_HEALTH_TO_TRIGGER_DIM = get_piecewise_dim(
    0, _HEALTH_TO_TRIGGER_MAX, _HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD
)

_MOVE_PREV_COUNT_DIM = get_piecewise_dim(0, MAX_MOVE_HISTORY, _LINEAR_SQRT_THRESHOLD)


# Health/block is not here: it is the combatants' health/block columns (health_block.py)


_CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
_OFF_MONSTER_NAME = _alloc("name", len(_IDX_OF_MONSTER_NAME))
_OFF_MONSTER_INTENT_DAMAGE = _alloc("intent_damage", _DAMAGE_DIM, 2)
_OFF_MONSTER_INTENT_DAMAGE_SCALAR = _alloc("intent_damage_scalar")
_OFF_MONSTER_INTENT_INSTANCES = _alloc("intent_instances")
_OFF_MONSTER_INTENT_DAMAGE_TOTAL = _alloc("intent_damage_total", _DAMAGE_TOTAL_DIM, 2)
_OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR = _alloc("intent_damage_total_scalar")
_OFF_MONSTER_INTENT_HAS_BLOCK = _alloc("intent_has_block")
_OFF_MONSTER_INTENT_HAS_BUFF = _alloc("intent_has_buff")
_OFF_MONSTER_INTENT_HAS_DEBUFF = _alloc("intent_has_debuff")
_OFF_MONSTER_HIT_NO_HEALTH_LOST = _alloc("hit_no_health_lost")
_OFF_MONSTER_HIT_IS_LETHAL = _alloc("hit_is_lethal")
_OFF_MONSTER_MAX_HEALTH_OHE = _alloc("max_health_ohe", _DIM_MAX_HEALTH_OHE, 2)
_OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc("max_health_magnitude")
_OFF_MONSTER_HEALTH_FRACTION = _alloc("health_fraction")
_OFF_MONSTER_POISON_TICKS_TO_KILL = _alloc("poison_ticks_to_kill", _POISON_TICKS_MAX)
_OFF_MONSTER_HEALTH_TO_TRIGGER = _alloc("health_to_trigger", _HEALTH_TO_TRIGGER_DIM, 2)
_OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR = _alloc("health_to_trigger_scalar")
_OFF_MONSTER_GOLD_STOLEN = _alloc("gold_stolen")
_OFF_MONSTER_INTENT_KIND = _alloc("intent_kind", len(_IDX_OF_INTENT_KIND))
_OFF_MONSTER_INTENT_HIDDEN = _alloc("intent_hidden")
_OFF_MONSTER_MOVE_SHOWN_IS_PREV_1 = _alloc("move_shown_is_prev_1")
_OFF_MONSTER_MOVE_SHOWN_IS_PREV_2 = _alloc("move_shown_is_prev_2")
_OFF_MONSTER_MOVE_PREV_1_IS_PREV_2 = _alloc("move_prev_1_is_prev_2")
_OFF_MONSTER_MOVE_PREV_COUNT = _alloc("move_prev_count", _MOVE_PREV_COUNT_DIM, 2)
_OFF_MONSTER_REPLAY_TARGET = _alloc("replay_target")
_OFF_MONSTER_HALF_DEAD = _alloc("half_dead")
_OFF_MONSTER_ULTIMATE_USED = _alloc("ultimate_used")

ENCODING_DIM_MONSTER = sum(w for _, w, _ in _CHANNEL_LAYOUT_MONSTER)

# Most nonzero columns a Monster row holds: one per one-hot or scalar, two per two-hot
ENCODING_DIM_MONSTER_SPARSE = sum(nonzeros for _, _, nonzeros in _CHANNEL_LAYOUT_MONSTER)


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
        _POISON_TICKS_MAX, a move may end it first (a Split's children hold no Poison, Anger and
        Haste clear it), or the Monster is half-dead, which no HP loss reaches.
    """
    if monster.half_dead:
        return None

    stacks_poison = stacks_of_modifier_kind.get(ModifierKind.Poison, 0)
    stacks_regen = stacks_of_modifier_kind.get(ModifierKind.Regeneration, 0)
    can_roll_debuffs_clear = (
        monster.name in _MONSTER_NAMES_HALF_HEALTH_ULTIMATE and not monster.ultimate_used
    )

    # Anger or Haste clears the Poison on its move, behind that turn's tick
    debuffs_clear_next = can_roll_debuffs_clear and any(
        isinstance(effect, EffectDebuffsClear)
        for effects in get_move_effects_shown(monster, ascension)
        for effect in effects
    )

    # No tick touches its block
    combatant_monster = make_combatant(
        EntityKind.MONSTER, monster.health, 0, stacks_of_modifier_kind
    )
    for ticks in range(1, min(stacks_poison, _POISON_TICKS_MAX) + 1):
        get_poison_tick(combatant_monster, stacks_poison - ticks + 1, ascension)
        if combatant_monster.health <= 0:
            return ticks
        
        if debuffs_clear_next or (
            ModifierKind.Splittable in stacks_of_modifier_kind
            and combatant_monster.health <= monster.health_max // 2
        ):
            return None

        # Below half, the roll behind its move is Anger or Haste
        debuffs_clear_next = (
            can_roll_debuffs_clear and combatant_monster.health < monster.health_max // 2
        )

        # Nemesis regains Intangible unless it still holds it
        combatant_monster.is_intangible = (
            monster.name == MonsterName.Nemesis and not combatant_monster.is_intangible
        )

        # Regen health
        combatant_monster.health = min(combatant_monster.health + stacks_regen, monster.health_max)

    return None


def encode_monster_into(
    monster: Monster,
    incoming: Incoming | None,
    outgoing_monster: list[tuple[int, bool]],
    ascension: int,
    np_own: np.ndarray,
    np_shared: np.ndarray,
) -> None:
    """Write a Monster's token: its own columns, then its health/block, slot and outgoing.

    Parameters
    ----------
    monster : Monster
        The Monster.
    incoming : Incoming | None
        What ending the turn now would cost (incoming.py); None when Runic Dome hides the intents.
    outgoing_monster : list[tuple[int, bool]]
        What each playable hand Card that hits it would take off it (card.py's
        `project_outgoing`): (HP lost, lethal).
    ascension : int
        The game's ascension, which picks the Monster's movesets and sets Invincible's cap on the
        health a Poison tick takes.
    np_own : np.ndarray
        (ENCODING_DIM_MONSTER,) float32. The Monster's own range of its token row, written in
        place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_own[_OFF_MONSTER_NAME + _IDX_OF_MONSTER_NAME[monster.name]] = 1.0
    encode_piecewise_into(
        monster.health_max,
        1,
        _HEALTH_MAX,
        _MAX_HEALTH_LINEAR_SQRT_THRESHOLD,
        np_own[_OFF_MONSTER_MAX_HEALTH_OHE:][:_DIM_MAX_HEALTH_OHE],
    )
    np_own[_OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_linear_norm(monster.health_max, SCALE_HEALTH)
    np_own[_OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    np_own[_OFF_MONSTER_REPLAY_TARGET] = float(monster.replay_target)
    np_own[_OFF_MONSTER_HALF_DEAD] = float(monster.half_dead)
    np_own[_OFF_MONSTER_ULTIMATE_USED] = float(monster.ultimate_used)

    # Poison's horizon
    stacks_of_modifier_kind = get_stacks_of_modifier_kind(monster.modifiers)
    poison_ticks = _poison_ticks_to_kill(monster, stacks_of_modifier_kind, ascension)
    if poison_ticks is not None:
        np_own[_OFF_MONSTER_POISON_TICKS_TO_KILL + poison_ticks - 1] = 1.0

    # HP to its trigger
    health_to_trigger = _health_to_trigger(
        monster.name,
        monster.health,
        monster.health_max,
        monster.ultimate_used,
        stacks_of_modifier_kind,
    )
    if health_to_trigger is not None:
        encode_piecewise_into(
            health_to_trigger,
            0,
            _HEALTH_TO_TRIGGER_MAX,
            _HEALTH_TO_TRIGGER_LINEAR_SQRT_THRESHOLD,
            np_own[_OFF_MONSTER_HEALTH_TO_TRIGGER:][:_HEALTH_TO_TRIGGER_DIM],
        )
        np_own[_OFF_MONSTER_HEALTH_TO_TRIGGER_SCALAR] = get_linear_norm(
            health_to_trigger, SCALE_HEALTH
        )

    # Stolen gold
    np_own[_OFF_MONSTER_GOLD_STOLEN] = get_linear_norm(monster.gold_stolen, SCALE_GOLD)

    # Intent
    intent = monster.intent
    if intent is None or incoming is None:
        np_own[_OFF_MONSTER_INTENT_HIDDEN] = 1.0
    else:
        damage = intent.damage or 0
        encode_piecewise_into(
            damage,
            0,
            _DAMAGE_MAX,
            _LINEAR_SQRT_THRESHOLD,
            np_own[_OFF_MONSTER_INTENT_DAMAGE:][:_DAMAGE_DIM],
        )

        total_damage = damage * (intent.instances or 1)
        np_own[_OFF_MONSTER_INTENT_DAMAGE_SCALAR] = get_linear_norm(damage, SCALE_HEALTH)
        np_own[_OFF_MONSTER_INTENT_INSTANCES] = (
            min(intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
        )
        encode_piecewise_into(
            total_damage,
            0,
            _DAMAGE_TOTAL_MAX,
            _LINEAR_SQRT_THRESHOLD,
            np_own[_OFF_MONSTER_INTENT_DAMAGE_TOTAL:][:_DAMAGE_TOTAL_DIM],
        )
        np_own[_OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR] = get_linear_norm(
            total_damage, SCALE_HEALTH
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

    # Move identity: the shown moves + the two before them
    moves_shown = monster.moves_shown
    move_history = monster.move_history
    if len(move_history) >= 1:
        np_own[_OFF_MONSTER_MOVE_SHOWN_IS_PREV_1] = float(move_history[-1] in moves_shown)
    if len(move_history) >= 2:
        np_own[_OFF_MONSTER_MOVE_SHOWN_IS_PREV_2] = float(move_history[-2] in moves_shown)
        np_own[_OFF_MONSTER_MOVE_PREV_1_IS_PREV_2] = float(move_history[-1] == move_history[-2])
    encode_piecewise_into(
        len(move_history),
        0,
        MAX_MOVE_HISTORY,
        _LINEAR_SQRT_THRESHOLD,
        np_own[_OFF_MONSTER_MOVE_PREV_COUNT:][:_MOVE_PREV_COUNT_DIM],
    )

    # Health & Block
    encode_shared_health_block_into(monster.health, monster.block, np_shared)

    # Stasis slot. Shared w/ Cards in the stasis pile
    encode_shared_stasis_slot_into(monster.slot, np_shared)

    # Outgoing: the best playable hand Card's. Shared w/ hand Cards
    if outgoing_monster:
        encode_shared_outgoing_into(
            monster.slot,
            max(health_lost_card for health_lost_card, _ in outgoing_monster),
            any(is_lethal_card for _, is_lethal_card in outgoing_monster),
            np_shared,
        )


# The catalog's movesets per (ascension, MonsterName), read once
_MOVESETS = {
    ascension: {template.name: template.movesets for template in get_monster_templates(ascension)}
    for ascension in range(ASCENSION_LEVEL + 1)
}


def _get_moves(monster: Monster, ascension: int, idx_move: int) -> list[MonsterMoveTemplate]:
    """Return what the move at an index of the Monster's moveset can be.

    Parameters
    ----------
    monster : Monster
        The Monster.
    ascension : int
        The game's ascension, which picks the Monster's movesets.
    idx_move : int
        The move's index in the Monster's moveset.

    Returns
    -------
    list[MonsterMoveTemplate]
        The move; while the Monster's spawn roll is hidden, the distinct moves its movesets hold
        there.
    """
    movesets = _MOVESETS[ascension][monster.name]
    if monster.moveset is not None:
        return [movesets[monster.moveset][idx_move]]

    # An entry the roll doesn't change comes once
    return list(dict.fromkeys(moveset[idx_move] for moveset in movesets))


def get_num_moves(monster: Monster, ascension: int) -> int:
    """Return the number of moves in the Monster's moveset.

    Parameters
    ----------
    monster : Monster
        The Monster.
    ascension : int
        The game's ascension, which picks the Monster's movesets.

    Returns
    -------
    int
        The number of moves, the same for every spawn roll.
    """
    return len(_MOVESETS[ascension][monster.name][0])


def get_move_effects(monster: Monster, ascension: int, idx_moves: Iterable[int]) -> list[Effect]:
    """Return the Effects of the moves at some indexes of the Monster's moveset.

    While the Monster's spawn roll is hidden, an index gives the Effects of every move a roll
    can put there.

    Parameters
    ----------
    monster : Monster
        The Monster.
    ascension : int
        The game's ascension, which picks the Monster's movesets.
    idx_moves : Iterable[int]
        The moves' indexes in the Monster's moveset.

    Returns
    -------
    list[Effect]
        Each move's Effects, in the indexes' order.
    """
    return [
        effect
        for idx_move in idx_moves
        for move in _get_moves(monster, ascension, idx_move)
        for effect in move.effects
    ]


def get_move_effects_shown(monster: Monster, ascension: int) -> tuple[list[Effect], list[Effect]]:
    """Return the Effects of the moves the Monster's intent shows.

    Parameters
    ----------
    monster : Monster
        The Monster.
    ascension : int
        The game's ascension, which picks the Monster's movesets.

    Returns
    -------
    effects_sure : list[Effect]
        The Effects every such move has.
    effects_possible : list[Effect]
        The Effects only some of them have.
    """
    moves = [
        move
        for idx_move in monster.moves_shown
        for move in _get_moves(monster, ascension, idx_move)
    ]
    if not moves:
        return [], []

    effects_sure = [
        effect for effect in moves[0].effects if all(effect in move.effects for move in moves)
    ]
    effects_possible = [
        effect for move in moves for effect in move.effects if effect not in effects_sure
    ]

    return effects_sure, effects_possible
