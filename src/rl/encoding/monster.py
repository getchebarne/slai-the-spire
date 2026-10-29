import math
from enum import StrEnum
from enum import auto

import numpy as np
from slai import Effect
from slai import GameState
from slai import IntentKind
from slai import ModifierKind
from slai import Monster
from slai import MonsterName
from slai import get_monster_templates

from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STEAL_MAX
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import MONSTER_MOVESET_MAX
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.types import make_channel_allocator


_MAP_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
_MAP_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
INTENT_BLOCK_KINDS = {IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff}
INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
INTENT_DEBUFF_KINDS = {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}

# Normalization caps, catalog-anchored
_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # Widest hit, Vulnerable-scaled
_DAMAGE_TOTAL_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)  # Widest turn, likewise
_INSTANCES_MAX = INTENT_INSTANCES_MAX
_HEALTH_MAX = HEALTH_MAX_ENTITY  # Champ 440 at A9+

# Thievery accumulation: Mug, Mug, Lunge — every attack before the Escape turn steals
_STOLEN_GOLD_MAX = 3 * GOLD_STEAL_MAX
_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
_DAMAGE_TOTAL_DIM = get_piecewise_dim(0, _DAMAGE_TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)


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
    HIT_FULLY_BLOCKED = auto()
    HIT_IS_LETHAL = auto()
    MAX_HEALTH_MAGNITUDE = auto()
    HEALTH_FRACTION = auto()
    POISON_IS_LETHAL = auto()  # Its next turn start's Poison tick kills it
    GOLD_STOLEN = auto()
    INTENT_KIND = auto()
    INTENT_HIDDEN = auto()  # Runic Dome: every intent channel and hit flag reads 0
    MOVE_CURRENT = auto()  # Index into the spawn-rolled moveset
    MOVE_PREV_1 = auto()  # One move back (move_history's last entry is the current move)
    MOVE_PREV_2 = auto()


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
OFF_MONSTER_HIT_FULLY_BLOCKED = _alloc(MonsterChannel.HIT_FULLY_BLOCKED)
OFF_MONSTER_HIT_IS_LETHAL = _alloc(MonsterChannel.HIT_IS_LETHAL)
OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc(MonsterChannel.MAX_HEALTH_MAGNITUDE)
OFF_MONSTER_HEALTH_FRACTION = _alloc(MonsterChannel.HEALTH_FRACTION)
OFF_MONSTER_POISON_IS_LETHAL = _alloc(MonsterChannel.POISON_IS_LETHAL)
OFF_MONSTER_GOLD_STOLEN = _alloc(MonsterChannel.GOLD_STOLEN)
OFF_MONSTER_INTENT_KIND = _alloc(MonsterChannel.INTENT_KIND, len(_MAP_INTENT_KIND))
OFF_MONSTER_INTENT_HIDDEN = _alloc(MonsterChannel.INTENT_HIDDEN)
OFF_MONSTER_MOVE_CURRENT = _alloc(MonsterChannel.MOVE_CURRENT, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_1 = _alloc(MonsterChannel.MOVE_PREV_1, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_2 = _alloc(MonsterChannel.MOVE_PREV_2, MONSTER_MOVESET_MAX)

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
    char_health: int,
    char_block: int,
    outgoing_damage: int | None,
    out: np.ndarray,
) -> None:
    out[OFF_MONSTER_NAME + _MAP_MONSTER_NAME[monster.name]] = 1.0
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)

    # Poison-is-lethal
    stacks_of_modifier_kind = {modifier.kind: modifier.stacks for modifier in monster.modifiers}
    poison_health_loss = stacks_of_modifier_kind.get(ModifierKind.Poison, 0)
    if ModifierKind.Buffer in stacks_of_modifier_kind:
        poison_health_loss = 0
    elif ModifierKind.Intangible in stacks_of_modifier_kind:
        poison_health_loss = min(poison_health_loss, 1)
    out[OFF_MONSTER_POISON_IS_LETHAL] = float(
        poison_health_loss > 0 and poison_health_loss >= monster.health
    )

    # Stolen gold
    out[OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, _STOLEN_GOLD_MAX)

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
        out[OFF_MONSTER_HIT_FULLY_BLOCKED] = float(total_damage <= char_block)
        out[OFF_MONSTER_HIT_IS_LETHAL] = float(outgoing_damage >= char_health + char_block)

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


def outgoing_damage(monsters: list[Monster]) -> int | None:
    # Runic Dome hides every intent, so the incoming damage is unknown
    if any(monster.intent is None for monster in monsters):
        return None

    return sum(_intent_total_damage(monster) for monster in monsters)


# Whole-moveset Effects per (ascension, MonsterName), collected once (the
# spawn-rolled variants are flattened)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, tuple[Effect, ...]]] = {}


def moveset_effects(ascension: int) -> dict[MonsterName, tuple[Effect, ...]]:
    by_name = _MOVESET_EFFECTS.get(ascension)
    if by_name is None:
        by_name = {
            template.name: tuple(
                effect
                for move in {move.name: move for move in template.moves}.values()
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
