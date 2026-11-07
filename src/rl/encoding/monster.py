import numpy as np
from slai import ModifierKind
from slai import Monster

from src.rl.constants import GOLD_STOLEN_MAX
from src.rl.constants import INVINCIBLE_STACKS_OF_ASCENSION
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.schema.monster import DAMAGE_MAX
from src.rl.schema.monster import DAMAGE_TOTAL_MAX
from src.rl.schema.monster import HEALTH_MAX
from src.rl.schema.monster import INSTANCES_MAX
from src.rl.schema.monster import INTENT_BLOCK_KINDS
from src.rl.schema.monster import INTENT_BUFF_KINDS
from src.rl.schema.monster import INTENT_DEBUFF_KINDS
from src.rl.schema.monster import LINEAR_SQRT_THRESHOLD
from src.rl.schema.monster import MAP_INTENT_KIND
from src.rl.schema.monster import MAP_MONSTER_NAME
from src.rl.schema.monster import OFF_MONSTER_GOLD_STOLEN
from src.rl.schema.monster import OFF_MONSTER_HALF_DEAD
from src.rl.schema.monster import OFF_MONSTER_HEALTH_FRACTION
from src.rl.schema.monster import OFF_MONSTER_HIT_FULLY_BLOCKED
from src.rl.schema.monster import OFF_MONSTER_HIT_IS_LETHAL
from src.rl.schema.monster import OFF_MONSTER_INTENT_DAMAGE
from src.rl.schema.monster import OFF_MONSTER_INTENT_DAMAGE_SCALAR
from src.rl.schema.monster import OFF_MONSTER_INTENT_DAMAGE_TOTAL
from src.rl.schema.monster import OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR
from src.rl.schema.monster import OFF_MONSTER_INTENT_HAS_BLOCK
from src.rl.schema.monster import OFF_MONSTER_INTENT_HAS_BUFF
from src.rl.schema.monster import OFF_MONSTER_INTENT_HAS_DEBUFF
from src.rl.schema.monster import OFF_MONSTER_INTENT_HIDDEN
from src.rl.schema.monster import OFF_MONSTER_INTENT_INSTANCES
from src.rl.schema.monster import OFF_MONSTER_INTENT_KIND
from src.rl.schema.monster import OFF_MONSTER_MAX_HEALTH_MAGNITUDE
from src.rl.schema.monster import OFF_MONSTER_MOVE_CURRENT
from src.rl.schema.monster import OFF_MONSTER_MOVE_PREV_1
from src.rl.schema.monster import OFF_MONSTER_MOVE_PREV_2
from src.rl.schema.monster import OFF_MONSTER_NAME
from src.rl.schema.monster import OFF_MONSTER_POISON_IS_LETHAL
from src.rl.schema.monster import OFF_MONSTER_REPLAY_TARGET
from src.rl.schema.monster import OFF_MONSTER_ULTIMATE_USED
from src.rl.schema.scalars import get_piecewise_bucket


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
    out[OFF_MONSTER_NAME + MAP_MONSTER_NAME[monster.name]] = 1.0
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, HEALTH_MAX)
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
        damage_bucket = get_piecewise_bucket(damage, 0, DAMAGE_MAX, LINEAR_SQRT_THRESHOLD)
        out[OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

        total_damage = _intent_total_damage(monster)
        out[OFF_MONSTER_INTENT_DAMAGE_SCALAR] = min(damage, DAMAGE_MAX) / DAMAGE_MAX
        out[OFF_MONSTER_INTENT_INSTANCES] = (
            min(intent.instances or 0, INSTANCES_MAX) / INSTANCES_MAX
        )
        total_damage_bucket = get_piecewise_bucket(
            total_damage, 0, DAMAGE_TOTAL_MAX, LINEAR_SQRT_THRESHOLD
        )
        out[OFF_MONSTER_INTENT_DAMAGE_TOTAL + total_damage_bucket] = 1.0
        out[OFF_MONSTER_INTENT_DAMAGE_TOTAL_SCALAR] = (
            min(total_damage, DAMAGE_TOTAL_MAX) / DAMAGE_TOTAL_MAX
        )
        out[OFF_MONSTER_INTENT_HAS_BLOCK] = float(intent.kind in INTENT_BLOCK_KINDS)
        out[OFF_MONSTER_INTENT_HAS_BUFF] = float(intent.kind in INTENT_BUFF_KINDS)
        out[OFF_MONSTER_INTENT_HAS_DEBUFF] = float(intent.kind in INTENT_DEBUFF_KINDS)
        out[OFF_MONSTER_HIT_FULLY_BLOCKED] = float(total_damage <= character_block)
        out[OFF_MONSTER_HIT_IS_LETHAL] = float(
            get_outgoing_damage >= character_health + character_block
        )

        # The intent category flags above miss Escape / Sleep / Stunned / Unknown
        out[OFF_MONSTER_INTENT_KIND + MAP_INTENT_KIND[intent.kind]] = 1.0

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
